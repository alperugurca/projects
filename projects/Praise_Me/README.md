## Praise Me (Övgü API)

FastAPI microservice that generates personalized praise-only responses using OpenAI.

### Quick start
- **Requirements**: Python 3.11+, OpenAI API key

Local:
```bash
pip install -r requirements.txt
echo OPENAI_API_KEY=your_openai_api_key > .env
uvicorn app:app --host 0.0.0.0 --port 8000
```

Docker:
```bash
echo OPENAI_API_KEY=your_openai_api_key > .env
docker compose up -d
```

### API
- POST `/chat` with JSON body:
```json
{ "query": "Her gün daha iyiyim demek doğru mu?" }
```
- Response:
```json
{ "status": "200", "message": "...övgü metni..." }
```
- Docs: `http://localhost:8000/docs`

### Config
- Required env: `OPENAI_API_KEY`

- Default model in code: `gpt-4.1-nano`

# ![Öv_Beni.png](Öv_Beni.png)
