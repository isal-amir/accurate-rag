1. Install dependencies

```
pip install -r requirements.txt
```

2. Create .env file covering all what in .env.example

3. Start the Qdrant server:

```
docker-compose up -d
```

3. Start the Streamlit server:

```
streamlit run app.py
```
