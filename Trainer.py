from torch.utils.data import DataLoader
from dataset import CustomDataset
from config import config
import os

class Trainer:
    def __init__(self, 
                img_width=160,
                img_height=60,
                max_len=4,
                ng=512,
                teacher_forcing_ratio=0.5,
                batch_size=32,
                lr= 3e-4,
                n_epoch=100,
                n_works=8,
                save_checkpoint_every = 5
                ):

        self.chars = config["chars"]
        self.root_path_to_train_images = config["root_path_to_train_images"]
        self.root_path_to_test_images = config["root_path_to_test_images"]

        self.path_to_train_gt_file = config["path_to_train_gt_file"]
        self.path_to_test_gt_file = config["path_to_test_gt_file"]

        self.ds_train = CustomDataset(
                self.root_path_to_train_images,
                self.path_to_train_gt_file,
                self.chars
                config["img_width"], 
                config["img_height"], 
                is_train=True
                )
        self.ds_test = CustomDataset(
                self.root_path_to_test_images,
                self.path_to_test_gt_file,
                chars
                img_width, 
                img_height, 
                is_train=False
                )

        self.tokenizer = ds_train.tokenizer

        self.train_loader = DataLoader(self.ds_train, batch_size=batch_size, shuffle=True, num_workers=n_works)
        self.test_loader = DataLoader(self.ds_test, batch_size=batch_size, shuffle=False, num_workers=n_works)

        self.model = OCR(img_width, img_height, nh, tokenizer.n_token,
                    max_len + 1, tokenizer.SOS_token, tokenizer.EOS_token).to(device=device)

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.NLLLoss().cuda()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train_batch(self, 
                    input_tensor, 
                    target_tensor, 
                    model, 
                    optimizer,
                    criterion, 
                    teacher_forcing_ratio, 
                    tokenizer):

        model.train()

        decoder_output = model(input_tensor, target_tensor, teacher_forcing_ratio)

        loss = 0

        optimizer.zero_grad()

        for i in range(decoder_output.size(1)):
            loss += criterion(decoder_output[:, i, :].squeeze(), target_tensor[:, i + 1])

        loss.backward()
        optimizer.step()

        target_tensor = target_tensor.cpu()
        decoder_output = decoder_output.cpu()

        prediction = torch.zeros_like(target_tensor)
        prediction[:, 0] = tokenizer.SOS_token
        for i in range(decoder_output.size(1)):
            prediction[:, i + 1] = decoder_output[:, i, :].squeeze().argmax(1)

        n_right = 0
        n_right_sentence = 0

        for i in range(prediction.size(0)):
            eq = prediction[i, 1:] == target_tensor[i, 1:]
            n_right += eq.sum().item()
            n_right_sentence += eq.all().item()

        return loss.item() / len(decoder_output), \
            n_right / prediction.size(0) / prediction.size(1), \
            n_right_sentence / prediction.size(0)

    def predict_batch(self, input_tensor, model):
        model.eval()
        decoder_output = model(input_tensor)

        return decoder_output


    def eval_batch(self, 
                    input_tensor, 
                    target_tensor, 
                    model, 
                    criterion, 
                    tokenizer):
        loss = 0

        decoder_output = predict_batch(input_tensor, model)

        for i in range(decoder_output.size(1)):
            loss += criterion(decoder_output[:, i, :].squeeze(), target_tensor[:, i + 1])

        target_tensor = target_tensor.cpu()
        decoder_output = decoder_output.cpu()

        prediction = torch.zeros_like(target_tensor)
        prediction[:, 0] = tokenizer.SOS_token

        for i in range(decoder_output.size(1)):
            prediction[:, i + 1] = decoder_output[:, i, :].squeeze().argmax(1)

        n_right = 0
        n_right_sentence = 0

        for i in range(prediction.size(0)):
            eq = prediction[i, 1:] == target_tensor[i, 1:]
            n_right += eq.sum().item()
            n_right_sentence += eq.all().item()

        return loss.item() / len(decoder_output), \
            n_right / prediction.size(0) / prediction.size(1), \
            n_right_sentence / prediction.size(0)

    def train_epoch(self):
        sum_loss_train = 0
        n_train = 0
        sum_acc = 0
        sum_sentence_acc = 0

        for bi, x, y in enumerate(tqdm(train_loader)):
            x = x.to(self.device)
            y = y.to(self.device)

            loss, acc, sentence_acc = train_batch(x, y, 
                                                self.model, 
                                                self.optimizer,
                                                self.criterion, 
                                                self.teacher_forcing_ratio, 
                                                self.max_len,
                                                self.tokenizer
                                                )

            sum_loss_train += loss
            sum_acc += acc
            sum_sentence_acc += sentence_acc

            n_train += 1

        return sum_loss_train / n_train, sum_acc / n_train, sum_sentence_acc / n_train

    def eval_epoch(self):
        sum_loss_eval = 0
        n_eval = 0
        sum_acc = 0
        sum_sentence_acc = 0

        for bi, batch in enumerate(tqdm(self.test_loader)):
            x, y = batch
            x = x.to(device=device)
            y = y.to(device=device)

            loss, acc, sentence_acc = eval_batch(x, y, model, crit, max_len, tokenizer)

            sum_loss_eval += loss
            sum_acc += acc
            sum_sentence_acc += sentence_acc

            n_eval += 1

        return sum_loss_eval / n_eval, sum_acc / n_eval, sum_sentence_acc / n_eval


    def run_engine(self):
        for epoch in range(n_epoch):
            train_loss, train_acc, train_sentence_acc = train_epoch()
            eval_loss, eval_acc, eval_sentence_acc = eval_epoch()

            print(f"Epoch: {epoch}")
            print(f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}")
            print(f"Eval Loss: {eval_loss:.4f} Eval Acc: {eval_acc:.4f}")

            if epoch % save_checkpoint_every == 0 and epoch > 0:
                print('saving checkpoint...')
                torch.save(model.state_dict(), './ckpts/time_%s_epoch_%s.pth' % (time.strftime('%Y-%m-%d_%H-%M-%S'), epoch))

    
if __name__ == "__main__":
    pass