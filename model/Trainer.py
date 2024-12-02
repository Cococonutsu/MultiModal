import os
import torch
import torch.nn as nn
import torch.optim
from tqdm import tqdm
import transformers

from model.T_processor import Text_Encoder
from model.A_processor import Audio_Encoder
from model.V_processor import Vision_Encoder

from utils.dataloader import get_dataloader
from utils.metrics import Metrics

from config import MOSI_config


class FusionModel(nn.Module):
    def __init__(self, config):
        super(FusionModel, self).__init__()
        self.text_encoder = Text_Encoder(config)
        self.audio_encoder = Audio_Encoder(config)
        self.vision_encoder = Vision_Encoder(config)

    def forward_text(self, texts_tokens, text_masks, labels, mode="single_cls"):
        return self.text_encoder(texts_tokens, text_masks, labels, mode)

    def forward_audio(self, audio_features, audio_masks, labels, mode="single_cls"):
        return self.audio_encoder(audio_features, audio_masks, labels, mode)

    def forward_vision(self, visions, vision_masks, labels):
        return self.vision_encoder(visions, vision_masks, labels)


class Trainer:
    def __init__(self, modal, config):
        self.modal = modal
        self.train_loader, self.test_loader, self.valid_loader = get_dataloader("MOSI", self.modal)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # 初始化 FusionModel
        self.model = FusionModel(MOSI_config).to(self.device)

        # 优化器和学习率调度器
        if self.modal == "text":
            lr, weight_deacy, amsgrad, num_warm_up = config.Text.lr, config.Text.weight_deacy, config.Text.amsgrad, config.Text.num_warm_up
        elif self.modal == "audio":
            lr, weight_deacy, amsgrad, num_warm_up = config.Audio.lr, config.Audio.weight_deacy, config.Audio.amsgrad, config.Audio.num_warm_up
        elif self.modal == "vision":
            lr, weight_deacy, amsgrad, num_warm_up = config.Vision.lr, config.Vision.weight_deacy, config.Vision.amsgrad, config.Vision.num_warm_up
        else:
            raise("参数传递错误，不属于三种模态中的任何一种")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_deacy, amsgrad=amsgrad)
        self.scheduler = transformers.optimization.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(num_warm_up * len(self.train_loader)),
            num_training_steps=MOSI_config.train_param.epochs * len(self.train_loader),
        )
        self.metrics = Metrics()


    def train(self):
        epoch = 0
        best_Loss = float("inf")
        patience_counter = 0
        early_stopping_patience = 10

        while patience_counter < early_stopping_patience:
            print("="*20, f"EPOCH:{epoch+1}", "="*20)
            self.model.train()
            epoch_loss = 0
            for batch in tqdm(self.train_loader):
                texts_tokens = batch["texts_tokens"].to(self.device)
                text_masks = batch["text_masks"].to(self.device)

                audio_features = batch["audio_features"].to(self.device)
                audio_masks = batch["audio_masks"].to(self.device)

                visions = batch["visions"].to(self.device)
                vision_masks = batch["vision_masks"].to(self.device)

                labels = batch["labels"].to(self.device)

                if self.modal == "text":
                    reg_result, train_loss = self.model.forward_text(texts_tokens, text_masks, labels)
                elif self.modal == "audio":
                    reg_result, train_loss = self.model.forward_audio(audio_features, audio_masks, labels)
                elif self.modal == "vision":
                    reg_result, train_loss = self.model.forward_vision(visions, vision_masks, labels)
                elif self.modal == "fusion":
                    pass

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += train_loss.item()

            avg_train_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}], Train Loss: {avg_train_loss:.4f}")

            # 测试部分
            eval_loss = self._eval()
            if eval_loss < best_Loss:
                best_Loss = eval_loss
                patience_counter = 0
                print(f"Validation loss improved to {eval_loss:.4f}")
            else:
                patience_counter += 1
                print(f"Validation loss did not improve. Patience counter: {patience_counter}/{early_stopping_patience}")

            epoch += 1
        print("Early stopping triggered.")

    def _eval(self):
        self.model.eval()
        total_loss = 0
        pred = []
        truth = []

        with torch.no_grad():
            for batch in tqdm(self.valid_loader):
                texts_tokens = batch["texts_tokens"].to(self.device)
                text_masks = batch["text_masks"].to(self.device)

                audio_features = batch["audio_features"].to(self.device)
                audio_masks = batch["audio_masks"].to(self.device)

                visions = batch["visions"].to(self.device)
                vision_masks = batch["vision_masks"].to(self.device)

                labels = batch["labels"].to(self.device)

                if self.modal == "text":
                    reg_result, valid_loss = self.model.forward_text(texts_tokens, text_masks, labels)
                elif self.modal == "audio":
                    reg_result, valid_loss = self.model.forward_audio(audio_features, audio_masks, labels)
                elif self.modal == "vision":
                    reg_result, valid_loss = self.model.forward_vision(visions, vision_masks, labels)
                elif self.modal == "fusion":
                    pass

                total_loss += valid_loss.item()
                pred.append(reg_result.view(-1))
                truth.append(labels)


            pred = torch.cat(pred).to(torch.device("cpu"))
            truth = torch.cat(truth).to(torch.device("cpu"))


            eval_results = self.metrics.eval_mosei_regression(truth, pred)
            eval_results['Loss'] = total_loss / len(self.valid_loader)

            log = '【%s】Pretarin_TrainAcc\n\tHas0_acc_2:%s\n\tHas0_F1_score:%s\n\tNon0_acc_2:%s\n\t' \
                'Non0_F1_score:%s\n\tMult_acc_5:%s\n\tMult_acc_7:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
                '------------------------------------------' % (
                    self.modal, eval_results['Has0_acc_2'], eval_results['Has0_F1_score'],
                    eval_results['Non0_acc_2'], eval_results['Non0_F1_score'], eval_results['Mult_acc_5'],
                    eval_results['Mult_acc_7'], eval_results['MAE'], eval_results['Corr'], round(eval_results["Loss"], 4))
            print(log)
        return eval_results["Loss"]

