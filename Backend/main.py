from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai.errors import APIError

app = FastAPI(title="MGP AI Advisor API")

# Allow CORS so the frontend HTML file can communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AdviceRequest(BaseModel):
    prompt: str
    api_key: str

@app.post("/api/advice")
def get_ai_advice(request_data: AdviceRequest):
    if not request_data.api_key:
        raise HTTPException(status_code=401, detail="API key is required")

    try:
        # Initialise le client officiel Google GenAI
        client = genai.Client(api_key=request_data.api_key)
        
        # Le modèle demandé par l'utilisateur
        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=request_data.prompt
        )
        
        return {"advice": response.text}
        
    except APIError as e:
        # Gestion des erreurs d'API Google (clé invalide, quotas, etc.)
        status = e.code if hasattr(e, 'code') else 500
        message = e.message if hasattr(e, 'message') else str(e)
        raise HTTPException(status_code=status, detail=f"Erreur API Google: {message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")