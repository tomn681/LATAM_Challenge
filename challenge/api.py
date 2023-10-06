import fastapi
import pandas as pd

from model import DelayModel

app = fastapi.FastAPI()

model = DelayModel()
model.load_from_weights("./challenge/MODEL.txt")

def read_file(
    json: dict
) -> pd.DataFrame:
    """
    Read file from Json.
    
    Args:
        json (dict): Raw data input
        
    Returns:
        data (pd.Dataframe): Processed data
    """    
    data = pd.DataFrame.from_dict(json["flights"])
    
    data = model.preprocess(data)
    
    return data
    
def check_input(
    json: dict
) -> list:
    """
    Check input values.
    
    Args:
        json (dict): Raw data input
        
    Returns:
        error (list): Non existing features
    """
    lst = [name + "_" + str(flight[name]) for flight in json["flights"] for name in flight]
    
    error = [element for element in lst if element not in ['OPERA_Aerolineas Argentinas', 
       'OPERA_Aeromexico', 'OPERA_Air Canada',
       'OPERA_Air France', 'OPERA_Alitalia', 'OPERA_American Airlines',
       'OPERA_Austral', 'OPERA_Avianca', 'OPERA_British Airways',
       'OPERA_Copa Air', 'OPERA_Delta Air', 'OPERA_Gol Trans',
       'OPERA_Grupo LATAM', 'OPERA_Iberia', 'OPERA_JetSmart SPA',
       'OPERA_K.L.M.', 'OPERA_Lacsa', 'OPERA_Latin American Wings',
       'OPERA_Oceanair Linhas Aereas', 'OPERA_Plus Ultra Lineas Aereas',
       'OPERA_Qantas Airways', 'OPERA_Sky Airline', 'OPERA_United Airlines',
       'TIPOVUELO_I', 'TIPOVUELO_N', 'MES_1', 'MES_2', 'MES_3', 'MES_4',
       'MES_5', 'MES_6', 'MES_7', 'MES_8', 'MES_9', 'MES_10', 'MES_11',
       'MES_12']]

    return error

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(json: dict) -> dict:

    error = check_input(json)
    if error:
        raise fastapi.HTTPException(status_code=400, 
                detail=f"Unknown features: {error}")

    data = read_file(json)
    
    return {
    	"predict": model.predict(data)
    }
