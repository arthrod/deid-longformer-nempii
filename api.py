# api.py - Clinical De-identification FastAPI Service
# Port 8001 on EC2 alongside CPT service (port 8000)

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time
import os

from labels import ID2LABEL, ENTITY_TYPES
from deid import ClinicalDeidentifier

# === Configuration ===
MODEL_PATH = os.environ.get("DEID_MODEL_PATH", "checkpoints/best_model")
MAX_TOKENS = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load Model (once at startup) ===
print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()
print(f"Model loaded: {model.num_parameters():,} parameters")

# === FastAPI App ===
app = FastAPI(
    title="Clinical De-identification API",
    description="PHI de-identification for clinical notes using Clinical-Longformer",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Request/Response Models ===
class DeidRequest(BaseModel):
    text: str = Field(..., description="Clinical note text to de-identify")
    return_entities: bool = Field(False, description="Include detected entities in response")
    seed: Optional[int] = Field(None, description="Random seed for reproducible replacements")

class DeidResponse(BaseModel):
    deidentified_text: str
    entities: Optional[list] = None
    token_count: int
    processing_time_ms: float

class ValidateRequest(BaseModel):
    text: str

class ValidateResponse(BaseModel):
    valid: bool
    token_count: int
    exceeds_limit: bool
    message: str

class BatchRequest(BaseModel):
    notes: list[str]
    return_entities: bool = False
    seed: Optional[int] = None

class BatchResponse(BaseModel):
    results: list[DeidResponse]
    total_processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    max_tokens: int

# === Core Inference Logic ===
def extract_entities(text: str) -> list[dict]:
    """Run model inference and extract entity spans."""
    
    # Tokenize
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        return_offsets_mapping=True,
        padding=True,
    )
    
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
    
    # Convert BILOU predictions to entity spans
    entities = []
    current_entity = None
    
    for idx, (pred_id, offsets) in enumerate(zip(predictions, offset_mapping)):
        # Skip special tokens
        if offsets[0] == 0 and offsets[1] == 0:
            continue
            
        label = ID2LABEL.get(pred_id, "O")
        
        if label == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
        
        tag, entity_type = label.split("-", 1)
        
        if tag == "B":  # Beginning
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "type": entity_type,
                "start": offsets[0],
                "end": offsets[1],
                "text": text[offsets[0]:offsets[1]]
            }
        elif tag == "I" and current_entity and current_entity["type"] == entity_type:
            # Inside - extend current entity
            current_entity["end"] = offsets[1]
            current_entity["text"] = text[current_entity["start"]:offsets[1]]
        elif tag == "L" and current_entity and current_entity["type"] == entity_type:
            # Last - close current entity
            current_entity["end"] = offsets[1]
            current_entity["text"] = text[current_entity["start"]:offsets[1]]
            entities.append(current_entity)
            current_entity = None
        elif tag == "U":  # Unit (single token entity)
            if current_entity:
                entities.append(current_entity)
            entities.append({
                "type": entity_type,
                "start": offsets[0],
                "end": offsets[1],
                "text": text[offsets[0]:offsets[1]]
            })
            current_entity = None
        else:
            # Invalid sequence - close and start new
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "type": entity_type,
                "start": offsets[0],
                "end": offsets[1],
                "text": text[offsets[0]:offsets[1]]
            }
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

def deidentify_text(text: str, entities: list[dict], seed: Optional[int] = None) -> str:
    """Replace detected entities with fake data."""
    
    deid = ClinicalDeidentifier(seed=seed)
    deid.reset_cache()
    
    # Sort entities by start position (reverse) for safe replacement
    sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)
    
    result = text
    for entity in sorted_entities:
        replacement = deid.replace(entity["text"], entity["type"])
        result = result[:entity["start"]] + replacement + result[entity["end"]:]
    
    return result

# === API Endpoints ===
@app.get("/deid/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device=DEVICE,
        max_tokens=MAX_TOKENS,
    )

@app.post("/deid/validate", response_model=ValidateResponse)
async def validate_text(request: ValidateRequest):
    """Pre-flight validation for text size."""
    tokens = tokenizer.encode(request.text, add_special_tokens=True)
    token_count = len(tokens)
    exceeds = token_count > MAX_TOKENS
    
    return ValidateResponse(
        valid=not exceeds,
        token_count=token_count,
        exceeds_limit=exceeds,
        message=f"Text has {token_count} tokens" + (f" (exceeds {MAX_TOKENS} limit)" if exceeds else ""),
    )

@app.post("/deid/process", response_model=DeidResponse)
async def process_text(request: DeidRequest):
    """De-identify a single clinical note."""
    start_time = time.time()
    
    # Validate length
    tokens = tokenizer.encode(request.text, add_special_tokens=True)
    if len(tokens) > MAX_TOKENS:
        raise HTTPException(
            status_code=400,
            detail=f"Text exceeds {MAX_TOKENS} token limit ({len(tokens)} tokens). Use chunking for longer documents."
        )
    
    # Extract entities
    entities = extract_entities(request.text)
    
    # De-identify
    deidentified = deidentify_text(request.text, entities, request.seed)
    
    processing_time = (time.time() - start_time) * 1000
    
    return DeidResponse(
        deidentified_text=deidentified,
        entities=entities if request.return_entities else None,
        token_count=len(tokens),
        processing_time_ms=round(processing_time, 2),
    )

@app.post("/deid/batch", response_model=BatchResponse)
async def process_batch(request: BatchRequest):
    """De-identify multiple clinical notes."""
    start_time = time.time()
    
    if len(request.notes) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100 notes")
    
    results = []
    for i, note in enumerate(request.notes):
        note_start = time.time()
        tokens = tokenizer.encode(note, add_special_tokens=True)
        
        if len(tokens) > MAX_TOKENS:
            raise HTTPException(
                status_code=400,
                detail=f"Note {i} exceeds {MAX_TOKENS} token limit"
            )
        
        entities = extract_entities(note)
        # Use seed + index for reproducibility across batch
        seed = request.seed + i if request.seed else None
        deidentified = deidentify_text(note, entities, seed)
        
        results.append(DeidResponse(
            deidentified_text=deidentified,
            entities=entities if request.return_entities else None,
            token_count=len(tokens),
            processing_time_ms=round((time.time() - note_start) * 1000, 2),
        ))
    
    return BatchResponse(
        results=results,
        total_processing_time_ms=round((time.time() - start_time) * 1000, 2),
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
