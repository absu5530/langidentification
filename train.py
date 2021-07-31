import fasttext

model = fasttext.train_supervised("train_augmented.txt", dim=50, minn=2, maxn=4, epoch=25, loss='hs')
print(model.test('valid_augmented.txt'))
model.save_model("langdetect_augmented.bin")
