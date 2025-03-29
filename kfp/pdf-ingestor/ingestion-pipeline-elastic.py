import os
from typing import List, NamedTuple

import kfp
from kfp import dsl, kubernetes, compiler
from kfp.dsl import Artifact, Input, Output

@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "langchain-community==0.3.8",
        "langchain==0.3.8",
        "pypdf==4.0.2",
        "tqdm==4.66.2",
    ],
)
def format_documents(splits_artifact: Output[Artifact]):
    import json
    import os
    from tqdm import tqdm
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print("📄 Starting PDF ingestion...")

    pdf_dir = "/data/pdfs"
    chunk_size = 2048
    chunk_overlap = 256

    all_splits = []

    for filename in tqdm(os.listdir(pdf_dir)):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            doc_id = os.path.splitext(filename)[0]

            loader = PyPDFLoader(filepath)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            splits = splitter.split_documents(docs)

            for i, split in enumerate(splits):
                split.metadata["source"] = filename
                split.metadata["chunk_index"] = i
                split.page_content = f"Document: {filename}\n\n{split.page_content}"
                all_splits.append({
                    "page_content": split.page_content,
                    "metadata": split.metadata
                })

    print(f"✅ Finished splitting {len(all_splits)} chunks from PDFs.")

    with open(splits_artifact.path, "w") as f:
        f.write(json.dumps([{
            "index_name": "pdf_ingestion_index",
            "splits": all_splits
        }]))

@dsl.component(
    base_image="image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/minimal-gpu:2024.2",
    packages_to_install=[
        "langchain-community==0.3.8",
        "langchain==0.3.8",
        "weaviate-client==3.26.2",
        "elastic-transport==8.15.1",
        "elasticsearch==8.16.0",
        "langchain-elasticsearch==0.3.0",
        "sentence-transformers==2.4.0",
        "einops==0.7.0",
    ],
)
def ingest_documents(input_artifact: Input[Artifact]):
    import json
    import os

    from elasticsearch import Elasticsearch
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from langchain_elasticsearch import ElasticsearchStore

    # Reading artifact from previous step into variable
    document_splits = []
    with open(input_artifact.path) as input_file:
        splits_artifact = input_file.read()
        document_splits = json.loads(splits_artifact)

    es_user = os.environ.get("ES_USER")
    es_pass = os.environ.get("ES_PASS")
    es_host = os.environ.get("ES_HOST")

    if not es_user or not es_pass or not es_host:
        print("Elasticsearch config not present. Check host, port and api_key")
        exit(1)

    # Iniatilize Elastic client
    es_client = Elasticsearch(es_host, 
                              basic_auth=(es_user, es_pass), 
                              request_timeout=30, 
                              verify_certs=False)

    # # Health check for elastic client connection
    print(f"Elastic Client status: {es_client.health_report()}")

    def ingest(index_name, splits):
        # Here we use Nomic AI's Nomic Embed Text model to generate embeddings
        # Adapt to your liking
        model_kwargs = {"trust_remote_code": True, "device": "cuda"}
        embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1",
            model_kwargs=model_kwargs,
            show_progress=True,
        )

        if es_client.indices.exists(index=index_name.lower()):
            print(f"Index {index_name.lower()} already exists, skipping ingestion.")
            return

        db = ElasticsearchStore(
            index_name=index_name.lower(),  # index names in elastic must be lowercase
            embedding=embeddings,
            es_connection=es_client,
        )

        print(f"Uploading document to collection {index_name}")
        db.add_documents(splits)

    for index_name, splits in document_splits:
        documents = [Document(page_content=split["page_content"], metadata=split["metadata"]) for split in splits]
        ingest(index_name=index_name, splits=documents)

    print("Finished!")

@dsl.pipeline(name="PDF Ingestion Pipeline")
def ingestion_pipeline():
    pvc1 = kubernetes.CreatePVC(
        pvc_name='pdf-storage',
        access_modes=['ReadWriteOnce'],
        size='5Gi',
    )    

    format_docs_task = format_documents()
    format_docs_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")

    kubernetes.mount_pvc(
        format_docs_task,
        pvc_name=pvc1.outputs['name'],
        mount_path='/data/pdfs',
    )

    ingest_docs_task = ingest_documents(input_artifact=format_docs_task.outputs["splits_artifact"])
    ingest_docs_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")

    kubernetes.mount_pvc(
        ingest_docs_task,
        pvc_name=pvc1.outputs['name'],
        mount_path='/data/pdfs',
    )
    kubernetes.use_secret_as_env(
        ingest_docs_task,
        secret_name="elasticsearch-es-elastic-user",
        secret_key_to_env={"elastic": "ES_PASS"},
    )
    ingest_docs_task.set_env_variable("ES_HOST", "http://elasticsearch-es-http:9200")
    ingest_docs_task.set_env_variable("ES_USER", "elastic")

    kubernetes.add_toleration(format_docs_task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")
    kubernetes.add_toleration(ingest_docs_task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")


if __name__ == "__main__":
    KUBEFLOW_ENDPOINT = os.getenv("KUBEFLOW_ENDPOINT")
    print(f"Connecting to kfp: {KUBEFLOW_ENDPOINT}")
    sa_token_path = "/run/secrets/kubernetes.io/serviceaccount/token"  # noqa: S105
    if os.path.isfile(sa_token_path):
        with open(sa_token_path) as f:
            BEARER_TOKEN = f.read().rstrip()
    else:
        BEARER_TOKEN = os.getenv("BEARER_TOKEN")

    sa_ca_cert = "/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"
    if os.path.isfile(sa_ca_cert):
        ssl_ca_cert = sa_ca_cert
    else:
        ssl_ca_cert = None

    client = kfp.Client(
        host=KUBEFLOW_ENDPOINT,
        existing_token=BEARER_TOKEN,
        ssl_ca_cert=None,
    )
    result = client.create_run_from_pipeline_func(
        ingestion_pipeline,
        experiment_name="pdf_ingestion",
        # enable_caching=False
    )
