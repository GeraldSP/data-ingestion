import os
from typing import List, NamedTuple

import kfp
from kfp import dsl, kubernetes
from kfp.dsl import Artifact, Input, Output

# 📥 Paso 1: Descargar PDFs desde MinIO
@dsl.component(
    base_image="python:3.11",
    packages_to_install=["boto3==1.34.46"],
)
def download_pdfs_from_minio():
    import os
    import boto3

    minio_endpoint = os.getenv("MINIO_ENDPOINT")
    minio_bucket = os.getenv("MINIO_BUCKET")
    minio_access_key = os.getenv("MINIO_ACCESS_KEY")
    minio_secret_key = os.getenv("MINIO_SECRET_KEY")

    pdf_dir = "/data/pdfs"
    os.makedirs(pdf_dir, exist_ok=True)

    s3 = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
    )

    objects = s3.list_objects_v2(Bucket=minio_bucket).get("Contents", [])
    for obj in objects:
        if obj["Key"].endswith(".pdf"):
            local_path = os.path.join(pdf_dir, os.path.basename(obj["Key"]))
            s3.download_file(minio_bucket, obj["Key"], local_path)
            print(f"✅ Downloaded {obj['Key']} to {local_path}")


# 📄 Paso 2: Formatear y dividir PDFs
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


# 📦 Paso 3: Ingestar en Elasticsearch
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

    es_client = Elasticsearch(
        es_host,
        basic_auth=(es_user, es_pass),
        request_timeout=30,
        verify_certs=False
    )

    print(f"Elastic Client status: {es_client.health_report()}")

    def ingest(index_name, splits):
        model_kwargs = {"trust_remote_code": True, "device": "cuda"}
        embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1",
            model_kwargs=model_kwargs,
            show_progress=True,
        )

        db = ElasticsearchStore(
            index_name=index_name.lower(),
            embedding=embeddings,
            es_connection=es_client,
        )

        print(f"Uploading document to collection {index_name}")
        db.add_documents(splits)

    for index_name, splits in document_splits:
        documents = [Document(page_content=split["page_content"], metadata=split["metadata"]) for split in splits]
        ingest(index_name=index_name, splits=documents)

    print("Finished!")


# 🚀 Pipeline principal
@dsl.pipeline(name="PDF Ingestion Pipeline")
def ingestion_pipeline():
    download_task = download_pdfs_from_minio()

    format_docs_task = format_documents()
    format_docs_task.after(download_task)
    format_docs_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")

    ingest_docs_task = ingest_documents(input_artifact=format_docs_task.outputs["splits_artifact"])
    ingest_docs_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")

    kubernetes.use_secret_as_env(
        download_task,
        secret_name="minio-secret",
        secret_key_to_env={
            "MINIO_ENDPOINT": "MINIO_ENDPOINT",
            "MINIO_BUCKET": "MINIO_BUCKET",
            "MINIO_ACCESS_KEY": "MINIO_ACCESS_KEY",
            "MINIO_SECRET_KEY": "MINIO_SECRET_KEY",
        },
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
    sa_token_path = "/run/secrets/kubernetes.io/serviceaccount/token"
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
        experiment_name="s3_pdf_ingestion",
    )