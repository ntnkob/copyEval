# Local installation

## Requirements
- OS (Linux/MacOS)
- Python package manager (This guide will be based on uv, but pip (should be) fine)
	
## Steps
1. Create an environment ```uv venv <name>```
2. Follow on-screen instructions to activate the environment or ```source <env_name>/bin/activate```
3. Clone the copyEval repository to your system
	a. In Bitbucket, you must create your API access key first (at least give it permission to read repository) (Create an API token | Bitbucket Cloud | Atlassian Support)
	b. ```git clone https://x-bitbucket-api-token-auth:<API_key>@bitbucket.org/gbdi-system/copyeval.git ./<folder_name>```
4. Navigate to (cd into) your ```<folder_name>``` from 3.
5. ```uv pip install -e .```

## Additional notes:
### Enabling Jupyter notebook
1. Run ```uv pip install ipykernel```

### Enabling RAGAS
1. Install RAGAS, ```uv pip install ragas```
2. Install scikit-learn, ```uv pip install scikit-learn```
3. Install mteb, ```uv pip install mteb```
4. Downgrade langchain-community, ```uv pip install "langchain-community<0.4.2"```
5. Install torch (cpu version is enough), ```uv pip install torch --index-url https://download.pytorch.org/whl/cpu```
