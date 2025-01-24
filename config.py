class Config:

    RANDOM_SEED = 33
    N_TIME_STEPS = 125   # 50 records in each sequence
    N_FEATURES = 2      # mag,hr,roi_Ratio,output
    step = 100           # window overlap = 50 -10 = 40  (80% overlap)
    N_CLASSES = 3       # class labels