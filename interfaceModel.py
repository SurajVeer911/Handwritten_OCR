import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, img: np.ndarray):
        img = cv2.resize(img, self.input_shapes[0][1:3][::-1])
        imgPred = np.expand_dims(img, axis=0).astype(np.float32)
        pred = self.model.run(self.output_names, {self.input_names[0]: imgPred})[0]
        text = ctc_decoder(pred, self.char_list)[0]
        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs
    # model_name = "2024-05-12--1844" # low performance, cer=17%
    model_name="202301111911" # high performance, cer=7%
    if(len(model_name) == 0):
          raise SystemExit("No model selected.")
    configs = BaseModelConfigs.load(f"Model/{model_name}/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    df = pd.read_csv(f"Model/{model_name}/val.csv").values.tolist()
    # df = df[:100]
    accCER = []
    for imgPath, label in tqdm(df):
        img = cv2.imread(imgPath.replace("\\", "/"))

        prediction_text = model.predict(img)

        cer = get_cer(prediction_text, str(label))
        print(f"Image: {imgPath}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

        accCER.append(cer)
 
        # uncomment when need to see the image as well
        # img = cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4))
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    print(f"Average CER: {np.average(accCER)}")