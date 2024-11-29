import os
import torch
import torch.nn as nn
import torch.optim
from tqdm import tqdm
import transformers

from model.A_processor import Audio_Encoder
from utils.dataloader import get_dataloader
from utils.metrics import Metrics

from config import MOSI_config




class Audio_trainer():
    def __init__(self):
        self.train_loader, self.test_loader, self.valid_loader = get_dataloader("MOSI")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = Audio_Encoder(MOSI_config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=MOSI_config.train_param.lr, weight_decay=1e-3)
        self.scheduler = transformers.optimization.get_linear_schedule_with_warmup(self.optimizer,
                                                                   num_warmup_steps=int(
                                                                       MOSI_config.train_param.num_warm_up * (len(self.train_loader))),
                                                                   num_training_steps= MOSI_config.train_param.epochs * len(self.train_loader), )
        self.metrics = Metrics()


    def train(self):
        
        epoch = 0
        while True:
            #======================训练部分======================
            print("="*20, f"EPOCH:{epoch+1}", "="*20)
            self.model.train()
            epoch_loss = 0
            for batch in tqdm(self.train_loader):
                audio_features = batch["audio_features"].to(self.device)
                audio_masks = batch["audio_masks"].to(self.device)
                labels = batch["labels"].to(self.device)


                
                reg_result, train_loss = self.model(audio_features, audio_masks, labels)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                epoch_loss += train_loss.item()


            avg_train_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}], Train Loss: {avg_train_loss:.4f}")


            #======================测试部分=====================
            result, result_loss = self._eval()
            log = 'visionPretarin_TrainAcc\n\tEpoch:%d\n\tHas0_acc_2:%s\n\tHas0_F1_score:%s\n\tNon0_acc_2"%s\n\t' \
              'Non0_F1_score:%s\n\tMult_acc_5:%s\n\tMult_acc_7:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
              '------------------------------------------' % (
                  epoch, result['Has0_acc_2'], result['Has0_F1_score'],
                  result['Non0_acc_2'], result['Non0_F1_score'], result['Mult_acc_5'],
                  result['Mult_acc_7'], result['MAE'], result['Corr'], round(result_loss,4))
            print(log)

            epoch += 1

    def _eval(self):
        self.model.eval()  
        total_loss = 0
        pred = []
        truth = []

        with torch.no_grad(): 
            for batch in tqdm(self.valid_loader):
                audio_features = batch["audio_features"].to(self.device)
                audio_masks = batch["audio_masks"].to(self.device)
                labels = batch["labels"].to(self.device)


                
                reg_result, valid_loss = self.model(audio_features, audio_masks, labels)
                total_loss += valid_loss.item()

                pred.append(reg_result.view(-1))
                truth.append(labels)
                # 计算损失

            pred = torch.cat(pred).to(torch.device("cpu"))
            truth = torch.cat(truth).to(torch.device("cpu"))
            evaL_results = self.metrics.eval_mosei_regression(truth, pred)
            evaL_results['Loss'] = total_loss / len(self.valid_loader)
        return evaL_results, evaL_results["Loss"]