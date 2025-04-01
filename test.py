import joblib
import os
import logging
import pickle
logging.basicConfig(filename="debug.log", level=logging.DEBUG, filemode="a")
logger = logging.getLogger(__name__)
with open("datasets/fertilizer prediction/fert_1/label_encoders_1.pkl", "rb") as f:
            label_encoder_fert_1 = pickle.load(f)
            logger.debug(f"Type of label_encoder_fert_1 after loading: {type(label_encoder_fert_1)}")
            if not isinstance(label_encoder_fert_1, dict):
                raise ValueError("label_encoder_fert_1 is not a dictionary")