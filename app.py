from typing import Dict, Literal, List
from pydantic import Field, BaseModel
from fastapi import FastAPI

import numpy as np
import pandas as pd
from itertools import product

import pickle

# options
co2_cost = True

class Powerplant(BaseModel):
    name: str
    type: Literal['gasfired', 'turbojet', 'windturbine']
    efficiency: float = Field(gt=0., le=1.)
    pmin: float = Field(ge=0.)
    pmax: float = Field(ge=0.)

class Fuels(BaseModel):
    gas: float = Field(alias="gas(euro/MWh)")
    kerosine: float = Field(alias="kerosine(euro/MWh)")
    co2: float = Field(alias="co2(euro/ton)")
    wind: float = Field(alias="wind(%)", ge=0., le=100.)

class Payload(BaseModel):
    load: float
    fuels: Fuels
    powerplants: List[Powerplant]

# functions for later use

def evaluate_cost(df_sub, load):
  """Caclulating the optimal cost of a set of power sources, assuming they are all switched on."""
  cost_min_total = df_sub['cost_min'].sum()
  rest_load = load - df_sub['pmin'].sum()
  pdelta_cum = pd.concat([pd.Series([0.],[-1]), df_sub['pdelta'].cumsum()])
  dyn_idx = pdelta_cum.gt(rest_load).idxmax()
  dyn_load = rest_load - pdelta_cum[dyn_idx-1]
  cost = cost_min_total + df_sub.loc[:dyn_idx-1,'cost_delta'].sum() + dyn_load * df_sub.loc[dyn_idx,'cost']
  return {
    'cost' : cost,
    'dyn_idx' : dyn_idx,
    'dyn_load' : dyn_load
  }

app = FastAPI()

@app.post("/productionplan/")
async def productionplan(payload : Payload, co2_cost : bool = True):
    with open('dump.pickle', 'wb') as file:
        pickle.dump(payload, file)
    load = payload.load
    fuels = payload.fuels
    powerplants = payload.powerplants
    num_powerplants = len(powerplants)
    
    # fuel cost per MWh (input)
    fuel_cost = {
    'gasfired' : fuels.gas + co2_cost*0.3*fuels.co2,
    'turbojet' : fuels.kerosine + co2_cost*0.3*fuels.co2,
    'windturbine' : 0.0
    }

    df = pd.DataFrame({i:dict(x) for i,x in enumerate(powerplants)}).transpose()
    wind_type = df['type'] == 'windturbine'
    df.loc[wind_type, 'pmin'] = df.loc[wind_type, 'pmax'] =\
        df.loc[wind_type, 'pmax']*(payload.fuels.wind/100)
    df['pdelta'] = df['pmax']-df['pmin']
    df['cost'] = df['type'].map(fuel_cost)/df['efficiency']
    df['cost_min'] = df['cost']*df['pmin']
    df['cost_delta'] = df['cost']*df['pdelta']
    df = df.sort_values('cost').reset_index(drop=True)

    if df['pmax'].sum() < payload.load:
        return {"error" : 
            "Provided power sources cannot produce enough to meet required load."}
    
    # Handling a load equal to 0
    if load == 0:
        return [ {'name':name, 'load':0} for name in df['name'] ]
    
    # Logic to find optimal configuration 
    best_price = np.inf
    for subset in product([False,True],repeat=num_powerplants):
        # changing order in which cartesian product is listed
        subset = list(subset)[::-1] 
        
        df_sub = df[subset].reset_index()
        
        # stop evaluation if the subset will produce too less or too much power
        if df_sub['pmin'].sum() > load:
            continue
        if df_sub['pmax'].sum() < load:
            continue

        config = evaluate_cost(df_sub, load)
        if config['cost'] < best_price:
            best_config = config
            best_price = best_config['cost']
            best_subset = subset
            df_optimal = df_sub.copy()

    if best_config is None:
        return {"error" : 
            "With provided power sources it is not possible to produce the exact amount required."}
    
    dyn_idx = best_config['dyn_idx']
    df_optimal['p'] = df_optimal['pmin'] # to be overwritten for certain values
    df_optimal.loc[:dyn_idx-1,'p'] = df_optimal['pmax']
    df_optimal.loc[dyn_idx, 'p'] = df_optimal.loc[dyn_idx, 'pmin'] + best_config['dyn_load']

    df['p'] = 0.
    df.loc[best_subset, 'p'] = df_optimal['p'].to_numpy()

    result = [{'name':name, 'p':p} for name, p in zip(df['name'], [round(p,1) for p in df['p']])]
    return result
