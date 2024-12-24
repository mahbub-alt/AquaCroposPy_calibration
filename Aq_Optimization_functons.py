# add soil data if there's internet, add flowering gdd

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent,IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import scipy.stats




def run_aquacrop_model( planting_date, harvest_date, smt, soil, wdf, sim_start ='2006/01/01', sim_end = '2022/12/30',
                       Emergence =80,  Maturity = 1700, MaxIrr =12, MaxIrrSeason =  600,MaxRooting =1409,HIstart =880,
                       Senescence = 1400, CCx = .96, WP = 33.7, Kcb =1.05, HI0 = 0.48, a_HI = 7.0, Zmax = 2.3):
    

             
    irr_mngt = IrrigationManagement(irrigation_method=1, SMT=smt, MaxIrrSeason=MaxIrrSeason, MaxIrr= MaxIrr)
        
    
    # Create Crop, InitialWaterContent, and IrrigationManagement instances
#     if crop is None:
      
    crop = Crop(c_name='MaizeGDD', 
                Name='MaizeGDD', planting_date=planting_date, harvest_date=harvest_date,GDDmethod = 3,
               Emergence =Emergence,  Maturity = Maturity, MaxRooting =MaxRooting,
                CCx =CCx, WP = WP, Kcb = Kcb, HI0 = HI0, a_HI = a_HI, Zmax = Zmax)
#     print('cc0', crop.CC0)
    crop.CGC = (
            np.log(
                (((0.98 * crop.CCx) - crop.CCx) * crop.CC0)
                / (-0.25 * (crop.CCx**2))
            )
        ) / (-(705 - crop.Emergence))
    
    
    
    tCD = crop.MaturityCD - crop.SenescenceCD
    if tCD <= 0:
        tCD = 1

    CCi = crop.CCx * (1 - 0.05 * (np.exp(((3.33 * crop.CDC_CD) / (crop.CCx + 2.29)) * tCD) - 1))
    if CCi < 0:
        CCi = 0

    tGDD = crop.Maturity - crop.Senescence
    if tGDD <= 0:
        tGDD = 5

    crop.CDC = ((crop.CCx + 2.29) * np.log((((CCi/crop.CCx) - 1) / -0.05) + 1)) / (3.33 * tGDD)
#     print(crop.Maturity)
#     print("CGC:", crop.CGC) 
#     print("CDC:",crop.CDC)
    initWC = InitialWaterContent(value=['FC'])
    
# MaxIrr= 6.5, MaxIrrSeason=600
    # Create AquaCropModel instance
    model = AquaCropModel(sim_start, sim_end, wdf, soil, crop,
                          initial_water_content=initWC, irrigation_management=irr_mngt, off_season=True)

    # Run the model
    model.run_model(till_termination=True)
    
    # Save results
    field_df = model._outputs.water_flux
    date_range = pd.date_range(start=sim_start, end=sim_end).strftime("%Y-%m-%d").tolist()
    field_df["Date"] = date_range
    field_df["ET_aqua"] = field_df.Es + field_df.Tr
    field_df["Date"] = pd.to_datetime(field_df["Date"])
    field_df.index = field_df["Date"]

    field_yld = model._outputs.final_stats
    gdd_df =model._outputs.crop_growth
    x= pd.Series([crop.Maturity], name='Maturity')
    
    return gdd_df, field_df, field_yld, x




# ---------------------------for All fields -------------------------


def run_aquacrop_model_for_fid(df, fid, sim_start='01/01', sim_end='12/30', smt=[55, 75, 45, 30], Emergence =80,
                               Maturity = 1700, MaxIrr =12,
                               CCx = .96, WP = 33.7, Kcb =1.05, HI0 = 0.48, a_HI = 7.0, Zmax = 2.3 ):

    # Subset the DataFrame for the current fid
    fid_df = df[df['fid'] == fid].copy()

    # Create WaterDataFrame (wdf) based on your data
    wdf = pd.DataFrame(fid_df[['MinTemp', 'MaxTemp', 'Precipitation', 'ReferenceET', 'Date']])
        # Default values for simulation parameters if not provided
   
 # maximum irrigation df

    sam_df =pd.read_csv("Data/FieldData_AllFieldsCompiled-Annual.csv")
    sam_df= sam_df[sam_df['cropType'].str.lower() == "corn"]
    maxirrdf = sam_df.groupby("FieldID")["irrigation_mm"].max().reset_index()   

    if fid in maxirrdf['FieldID'].values:
        # Get the corresponding max irrigation season
        max_irr_season = maxirrdf[maxirrdf['FieldID'] == fid]['irrigation_mm'].values[0]
    else:
        # Set default max irrigation season to 550
        max_irr_season = 600
    
    irr_mngt = IrrigationManagement(irrigation_method=1, SMT=smt, MaxIrrSeason=max_irr_season, MaxIrr= MaxIrr)
    

#     add gdd df
    gdd_df =pd.read_csv("Data/all_fields_GDD.csv") 

    default_maturity = 1700
    default_max_rooting = 1409
    default_hI_start = 880
    default_senescence = 1400
    soil = Soil(soil_type='SiltLoam')
    
    # Define planting and harvest dates based on FieldID
    if fid.startswith("SW"):
        
        Emergence = gdd_df.loc[gdd_df["FieldID"]=="SW", "Emergence" ].item()
        Maturity =  gdd_df.loc[gdd_df["FieldID"]=="SW", "Maturity" ].item()
        planting_date = '05/10'
        harvest_date = '10/07'
        soil = Soil(soil_type = 'SiltClayLoam')
        
    elif fid.startswith("NB") or fid.startswith("NW"):
        
        Emergence = gdd_df.loc[gdd_df["FieldID"]=="NW", "Emergence" ].item()
        Maturity =  gdd_df.loc[gdd_df["FieldID"]=="NW", "Maturity" ].item()
        
        planting_date = '05/18'
        harvest_date = '10/17'
        
    elif fid.startswith("NC"):
        
        Emergence = gdd_df.loc[gdd_df["FieldID"]=="NC", "Emergence" ].item()
        Maturity =  gdd_df.loc[gdd_df["FieldID"]=="NC", "Maturity" ].item()
        
        planting_date = '05/07'
        harvest_date = '10/12'
        
    elif fid.startswith("WC"):
        Emergence = gdd_df.loc[gdd_df["FieldID"]=="WC", "Emergence" ].item()
        Maturity =  gdd_df.loc[gdd_df["FieldID"]=="WC", "Maturity" ].item()
        
        planting_date = '05/16'
        harvest_date = '10/11'
        
    else:
        # Default planting and harvest dates if FieldID does not match any condition
        planting_date = '05/08'
        harvest_date = '10/07'
        
    percentage_change = (Maturity  - default_maturity) / default_maturity * 100
    MaxRooting = default_max_rooting + (percentage_change / 100) * default_max_rooting
#     print('maxrooting:',  MaxRooting)
    HIstart = default_hI_start + (percentage_change / 100) * default_hI_start
    Senescence = default_senescence + (percentage_change / 100) * default_senescence

    # Run AquaCrop model for the current fid
    gdd_df, field_df, field_yld, x = run_aquacrop_model(planting_date, harvest_date,smt =smt, soil =soil, wdf=wdf,                                                                     sim_start = sim_start, sim_end = sim_end, Emergence = Emergence,
                                                       Maturity = Maturity, MaxIrrSeason = max_irr_season,
                                                       MaxRooting = MaxRooting, HIstart = HIstart, Senescence = Senescence,
                                                       CCx =CCx, WP = WP, Kcb = Kcb, HI0 = HI0, a_HI = a_HI, Zmax = Zmax 
                                                       )
    # Add fid column to identify the data
    field_df['FieldID'] = fid
    field_yld['FieldID'] = fid
    gdd_df['FieldID'] = fid
    x['FieldID'] = fid
    
    return gdd_df, field_df, field_yld, x



                          # ********************************************
def for_objf (smt =[50, 65, 40, 30], CCx = 0.96, WP = 33.7, Kcb =1.05, HI0 = 0.48, 
              a_HI = 7.0, Zmax = 2.3,no_et = True, train = True ):

    # run all fields aquacrop model simulation
    df = pd.read_csv('Data/Updated_all_fieldsclimate.csv')
    df["Date"] = df.Date.str[:8]
    df["Date"] = pd.to_datetime(df["Date"])
    df =df[~df['fid'].isin(['NW6', 'NW7'])]
#     df = df[~df["fid"].isin(["NW6", "NW7"]) & ~df["fid"].str.startswith("WC")]

    unique_fids = df['fid'].unique()
    # DataFrames to store results for all fids
    all_ET_df = pd.DataFrame()
    all_yld_df = pd.DataFrame()

    # Iterate over each fid and run AquaCrop model
    for fid in unique_fids:
        model, field_df, field_yld, x = run_aquacrop_model_for_fid(df, fid, 
                                                                   sim_start='2006/01/01', 
                                                                   sim_end='2023/12/30', 
                                                                   smt = smt , CCx = CCx, WP = WP, Kcb = Kcb,
                                                                   HI0 = HI0, a_HI = a_HI, Zmax = Zmax
                                                                   )

        # Append results to the aggregated DataFrames
        all_ET_df = all_ET_df.append(field_df, ignore_index=True)
        all_yld_df = all_yld_df.append(field_yld, ignore_index=True)

    all_yld_df["Year"] = all_yld_df["Harvest Date (YYYY/MM/DD)"].dt.year
    all_ET_df["Year"] = all_ET_df.Date.dt.year
    

    
    ydff =pd.read_excel("Data/All_reported_data_shuffled.xlsx")
        
    if no_et:
        simul_reported =pd.merge(ydff,all_yld_df, on=['FieldID', 'Year'], how='inner')
        print("100 prcnt_without_ET:",len(simul_reported ))
        test_df = "Not available as we are using full data"
        if train:
            simul_reported, test_df = train_test_split(simul_reported, test_size=0.30, random_state=42)
            print("train_len_no_ET:",len(simul_reported ))
            
    else:
        #     OpenET data
        GEEdf =pd.read_csv("Data/Full_year_ET2.csv")
        #         Yearly sum
        sim_et= all_ET_df.groupby(["FieldID" ,"Year"])["ET_aqua"].sum().reset_index()
        all_ET_df =pd.merge(GEEdf,  sim_et, on=['FieldID', 'Year'], how='inner')
        simul_reported =  pd.merge(ydff,all_yld_df, on=['FieldID', 'Year'], how='inner') 
        simul_reported  = pd.merge ( all_ET_df ,  simul_reported, on=['FieldID', 'Year'], how='inner')
        print("100 prcnt_with_ET:",len(simul_reported ))
        test_df = "Not available as we are using full data"
        if train:
            simul_reported, test_df = train_test_split(simul_reported, test_size=0.30, random_state=42)
            print("train_len_with_ET:",len(simul_reported ))
    
    return test_df, all_yld_df, simul_reported



def obj_func (param, no_et =True, train = True):
    
    """
    you can remove any parameter from "for_objf" function
    to avaoid calibration of that parameter. for that lb and up adjustment is necessary.
    """

    smt = [param[0], param[1], param[2], param[3]]
    CCx = param[4]
    WP = param[5] 
    Kcb = param[6]
    HI0 = param[7]
    a_HI = param[8]
    Zmax = param[7]

# , a_HI = a_HI, Zmax = Zmax
    print("smt:", smt)
    print("CCx:", CCx)
    print("WP:",  WP)
    print("Kcb:", Kcb)
    print("HI0:", HI0)
    print("a_HI:", a_HI)
    print("Zmax:",Zmax)
    
    all_ET_df, all_yld_df, df = for_objf(smt =smt, CCx =CCx, WP = WP, Kcb = Kcb,
                                                     HI0 = HI0, a_HI = a_HI, Zmax = Zmax,
                                                     no_et = no_et,train =train)
    
    

    df.rename(columns={"Seasonal irrigation (mm)": "Simulated_Irrigation", "Yield (tonne/ha)": "Simulated_Yield"}, inplace =True)
    
   # Step 1: Calculate residuals
    df["Irrigation_Residual"] = (df["Reported_Irrigation"] - df["Simulated_Irrigation"]).abs()
    df["Yield_Residual"] = (df["Reported_Yield"] - df["Simulated_Yield"]).abs()



    # Step 2: Normalize residuals using Min-Max scaling
    def min_max_scaling(series):
        return (series - series.min()) / (series.max() - series.min())

    df["Normalized_Irrigation_Residual"] = min_max_scaling(df["Irrigation_Residual"])
    df["Normalized_Yield_Residual"] = min_max_scaling(df["Yield_Residual"])


    # Step 3: Compute mean of normalized residuals
    mean_normalized_irrigation_residual = df["Normalized_Irrigation_Residual"].mean()
    mean_normalized_yield_residual = df["Normalized_Yield_Residual"].mean()

    if no_et:
        weight_yield = 6
        weight_irrigation =4


        # Objective function
        fitness = (
           weight_irrigation * mean_normalized_irrigation_residual +
           weight_yield * mean_normalized_yield_residual
        )
        
        # only_yield;   now we don't need any weight
        # fitness  = mean_normalized_yield_residual
        #only_irrig
        # fitness = mean_normalized_irrigation_residual

        
        
        print("Irrigation contribution:",weight_irrigation * mean_normalized_irrigation_residual)
        print("Yield contribution:",  weight_yield *mean_normalized_yield_residual)

    else:
        df["ET_Residual"] =  (df['Ensemble_ET'] - df['ET_aqua']).abs()
        df["Normalized_ET_Residual"] = min_max_scaling(df["ET_Residual"])
        mean_normalized_ET_residual = df["Normalized_ET_Residual"].mean()
        
        weight_yield = 5
        weight_irrigation =3
        weight_et  = 2

        # Objective function
        fitness = (
            weight_irrigation * mean_normalized_irrigation_residual +
            weight_yield * mean_normalized_yield_residual +
            weight_et * mean_normalized_ET_residual
        )

        # only_ET
        # fitness =  mean_normalized_ET_residual
        

        print("Irrigation contribution:",weight_irrigation * mean_normalized_irrigation_residual)
        print("Yield contribution:",  weight_yield *mean_normalized_yield_residual)
        print("ET contribution:",  weight_et * mean_normalized_ET_residual)

    print('loss:', fitness)
    print("------------------------")
    
    return fitness

    
