import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


#path= "C:/Users/mr_ar/Downloads/Projet+Mise+en+prod+-+home-credit-default-risk/"

#data
df=pd.read_csv('df_test_api.csv')
df["SK_ID_CURR"]=df["SK_ID_CURR"].convert_dtypes()
sk=df["SK_ID_CURR"]
df.index=sk
X=df.copy()
X.drop(columns=["SK_ID_CURR"],inplace=True) 
#model
model = joblib.load("model_api.joblib")
model.steps.pop(0)


# Initialize an instance of FastAPI
app = FastAPI()
class Item(BaseModel):
    ID: int
    


# Define the default route 
@app.get("/")
async def root():
    return {"message": "Modèle de scorring"}


@app.post("/predictions")
async def predictions(input:Item):

    prediction=model.predict(X[X.index == input.ID]).tolist()[0]
    score=model.predict_proba(X[X.index == input.ID])[:,1]
    #return int(score[0])
    return {"ID" : input.ID, "prediction": int(prediction), "score":(round(score[0],3))
            }

#lancement de l'application
if __name__ == "__main__":
    app.run(debug=True)
    