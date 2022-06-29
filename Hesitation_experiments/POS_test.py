import sentencepiece as spm
import stanza

"""s = spm.SentencePieceProcessor(model_file='/home/lina/Desktop/Stage/tokenizers/en_tokenization.model')
print(" ".join(x for x in s.encode('the writer has finished his work.', out_type=str)))
print(" ".join(x for x in s.encode('the writer has finished her work.', out_type=str)))
print(" ".join(x for x in s.encode('the writer finished his work.', out_type=str)))
print(" ".join(x for x in s.encode('the writer finished her work.', out_type=str)))
print(" ".join(x for x in s.encode('the author has finished his work.', out_type=str)))"""

print(s.decode("▁the ▁writer ▁has ▁finished ▁his ▁work .".split()))

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', package={'constituency': 'wsj_bert'})
doc = nlp('the writer has finished his work.')
print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for sent in doc.sentences for word in sent.words], sep='\n')

for sentence in doc.sentences:
    print(sentence.constituency)
