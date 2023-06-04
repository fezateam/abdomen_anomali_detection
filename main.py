from imp import reload
import cv2
import pydicom as dicom
from zipfile import ZipFile
from glob import glob
import os
import pandas as pd
import yaml
import numpy as np
from mmdet.apis import init_detector, inference_detector
from matplotlib import pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime
reload(logging)


class AbdomenLesionDetector:
    def __init__(self, yaml_file) -> None:
        yfile = open(yaml_file, "r")
        self.yfile = yaml.safe_load(yfile)

        self.DATASET_PATH = self.yfile["dataset_path"]
        self.EXTRACTED_DATASET_PATH = self.yfile["extracted_dataset_path"]
        self.CLASSES = self.yfile["classes"]
        self.XLS_FILE = self.yfile["xls_file"]
        self.YEDEK = self.yfile["yedek"]
        self.models = self.loadModels()

    def loadModels(self):
        anevdis = init_detector(
            self.yfile["abdominal aorta"]["config"], self.yfile["abdominal aorta"]["weight"], device='cuda:0')
        apandisit = init_detector(
            self.yfile["apandiks"]["config"], self.yfile["apandiks"]["weight"], device='cuda:0')
        bobrekureter = init_detector(
            self.yfile["böbrek-üreter-mesane"]["config"], self.yfile["böbrek-üreter-mesane"]["weight"], device='cuda:0')
        kolesistit = init_detector(
            self.yfile["safra kesesi"]["config"], self.yfile["safra kesesi"]["weight"], device='cuda:0')
        pankreatit = init_detector(
            self.yfile["pankreas"]["config"], self.yfile["pankreas"]["weight"], device='cuda:0')

        # models = {"safra kesesi": kolesistit, "pankreas": pankreatit,
        #           "böbrek-üreter-mesane": bobrekureter, "apandiks": apandisit, "abdominal aorta": anevdis}
        models = {self.yfile["classes"][0]: kolesistit, self.yfile["classes"][1]: pankreatit,
                  self.yfile["classes"][2]: bobrekureter, self.yfile["classes"][3]: apandisit, self.yfile["classes"][5]: anevdis}

        return models

    def err(self):
        print("class is not match any weights.")
    

    def zip2readable(self):
        os.makedirs(self.EXTRACTED_DATASET_PATH, exist_ok=True)
        dataset = glob(self.DATASET_PATH+"/*.zip", recursive=True)
        for rar in dataset:
            zfile = ZipFile(rar, "r")
            zfile.extractall(self.EXTRACTED_DATASET_PATH)

    def window_image(self, dcm, window_center, window_width):
        img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)
        return img

    def bbox2coord(self, bbox):
        bbox = list(map(lambda x: str(round(x)), bbox))
        xy1 = ",".join(bbox[:2])
        xy2 = ",".join(bbox[2:])
        return "-".join([xy1, xy2])

    def saveExcel(self, xls):
        save_path = self.yfile["save_excel"].rsplit("/")[:-1]
        save_path = "/".join(save_path)
        os.makedirs(save_path, exist_ok=True)
        if self.YEDEK == 0:
            xls.to_excel(self.yfile["save_excel"])
            self.YEDEK = self.yfile["yedek"]
            logging.info(f"{datetime.now()}: Saving successful.")
        self.YEDEK -= 1

    def saveLabeledImg(self, img, bbox, seri, kesit, cls):
        os.makedirs(self.yfile["save_image_path"]+f"/{cls}", exist_ok=True)
        bbox = list(map(lambda x: round(x), bbox))
        img = cv2.rectangle(
            img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
        name = self.yfile["save_image_path"]+f"/{cls}/{seri}_{kesit}.jpg"

        cv2.imwrite(name, img)

    def main(self):
        os.makedirs("./notpredicted",exist_ok=True)
        xls = pd.read_excel(self.XLS_FILE)
        xls_col = list(xls.columns)
        xls["Koordinatlar"] = None
        for indx, x in enumerate(tqdm(xls.values)):

            olgu = x[xls_col.index("Olgu Numarası")]
            seri = olgu.rsplit("/")[0]
            kesit = x[xls_col.index("Kesit Numarası")]
            cls = x[xls_col.index("Sınıf")]

            dcm_path = f"{self.EXTRACTED_DATASET_PATH}/{olgu}/{kesit}.dcm"
            dcm = dicom.dcmread(dcm_path)
            w = dcm["Rows"].value
            h = dcm["Columns"].value

            # w_center= dcm[0x0028, 0x1050].value
            # w_width= dcm[0x0028, 0x1051].value
            w_center = 50
            w_width = 400

            img = self.window_image(dcm, w_center, w_width)
            img = img.astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            try:
                result = inference_detector(
                    self.models.get(cls, self.err), img)
            except:
                logging.info(
                    f"{datetime.now()}: olgu {olgu}, kesit {kesit} {cls} cannot predicted.\n")
                notpred_txt= open("./notpredicted/a.txt","a")
                notpred_txt.write(f"{self.EXTRACTED_DATASET_PATH}/{olgu}/{kesit}.dcm\n")
                notpred_txt.close()
                continue

            bbox_list = []
            class_list = []
            score_list = []
            for e, res in enumerate(result):
                if res.shape[0] < 1:
                    continue
                pred = res[0]

                if pred[-1] < self.yfile[cls]["iou_threshold"]:
                    continue
                pred_box = pred[:-1]
                bbox_list.append(pred_box)
                class_list.append(self.yfile[cls]["patoloji"][e])
                score_list.append(pred[-1])

            if len(bbox_list) > 0:
                max_score = max(score_list)
                max_index = score_list.index(max_score)

                pa_class = class_list[max_index]
                bbox = bbox_list[max_index]
                self.saveLabeledImg(img, bbox, seri, kesit, cls)
                bbox = self.bbox2coord(bbox)
            else:
                pa_class = cls
                bbox = ""

            logging.info(
                f"{datetime.now()}: {indx} {olgu}/{kesit} is done. bbox: {bbox}\n")
            xls.loc[indx, "Koordinatlar"] = bbox
            self.saveExcel(xls)


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename=f'logs/log_{datetime.now()}.log', level=logging.INFO)
    logging.info(f'{datetime.now()}: Started\n')
    ald = AbdomenLesionDetector("./data.yaml")
    # ald.zip2readable()
    ald.main()
