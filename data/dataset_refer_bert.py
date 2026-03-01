import os
import warnings
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image

from bert.tokenization_bert import BertTokenizer

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

from refer.refer import REFER

class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.args = args
        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)

        self.max_tokens = getattr(args, "max_tokens", 20)

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        self.eval_mode = eval_mode
        self.use_gpg = getattr(args, "use_gpg", False)
        if self.use_gpg:
            try:
                import re
                import nltk
                from nltk.tokenize import word_tokenize
            except Exception as exc:
                raise RuntimeError(
                    "use_gpg requires nltk. Please install nltk and download the required models."
                ) from exc
            self._re = re
            self._nltk = nltk
            self._word_tokenize = word_tokenize
            # RRSIS-D target class vocabulary (from the paper definition)
            self.target_cls = {
                "airplane", "airport", "golf field", "expressway service area", "baseball field", "stadium",
                "ground track field", "storage tank", "basketball court", "chimney", "tennis court", "overpass",
                "train station", "ship", "expressway toll station", "dam", "harbor", "bridge", "vehicle",
                "windmill"
            }
            self.target_masks = []
            self.position_masks = []
        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            attentions_for_ref = []
            if self.use_gpg:
                target_for_ref = []
                position_for_ref = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                target_mask = [0] * self.max_tokens
                position_mask = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

                if self.use_gpg:
                    tokenized_sentence = self._word_tokenize(sentence_raw)

                    if self.args.dataset == "RRSIS-D":
                        for cls in self.target_cls:
                            if self._re.findall(cls, sentence_raw):
                                tokenized_cls = self._word_tokenize(cls)
                                nums_cls = len(tokenized_cls)
                                index = 0
                                for j, token in enumerate(tokenized_sentence):
                                    if self._re.findall(tokenized_cls[0], token):
                                        index = j
                                        break
                                target_mask[index + 1: index + nums_cls + 1] = [1] * nums_cls
                    else:
                        # Generic: use noun POS tags as target words
                        tagged = self._nltk.pos_tag(tokenized_sentence)
                        for j, (_, pos) in enumerate(tagged):
                            if pos.startswith("NN"):
                                # Ensure index is in range to avoid IndexError
                                if j + 1 < self.max_tokens:
                                    target_mask[j + 1] = 1

                    # Position phrase extraction
                    grammar = r"""
                    PP: {<IN><DT>?<JJ.*>?<NN>}
                        {<IN><DT>?<JJ.*>?<JJ>}
                        {<IN><DT>?<JJ.*><VBD>}
                    """
                    chunkr = self._nltk.RegexpParser(grammar)
                    tree = chunkr.parse(self._nltk.pos_tag(tokenized_sentence))
                    pp_phrases = []
                    for subtree in tree.subtrees():
                        if subtree.label() == 'PP':
                            pp_phrases.append(' '.join(word for word, pos in subtree.leaves()))

                    new_pp_phrase = []
                    for phrase in pp_phrases:
                        if not self._re.findall("of", phrase):
                            new_pp_phrase.append(phrase)

                    if len(new_pp_phrase) > 0:
                        for pp in new_pp_phrase:
                            tokenized_pos = self._word_tokenize(pp)
                            nums_pos = len(tokenized_pos)
                            index = 0
                            for j, token in enumerate(tokenized_sentence):
                                if tokenized_pos[0] == token:
                                    index = j
                                    break
                            position_mask[index + 1: index + nums_pos + 1] = [1] * nums_pos

                    # If no position phrase is found, fall back to attention_mask
                    if sum(position_mask) == 0:
                        position_mask = attention_mask[:]

                    # Ensure lengths match input_ids/attention_mask to avoid DataLoader stack errors
                    # (slice assignment may extend target_mask/position_mask beyond max_tokens)
                    target_mask = (target_mask + [0] * self.max_tokens)[:self.max_tokens]
                    position_mask = (position_mask + [0] * self.max_tokens)[:self.max_tokens]

                    target_for_ref.append(torch.tensor(target_mask).unsqueeze(0))
                    position_for_ref.append(torch.tensor(position_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)
            if self.use_gpg:
                self.target_masks.append(target_for_ref)
                self.position_masks.append(position_for_ref)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")

        ref = self.refer.loadRefs(this_ref_id)

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
            img, target = self.image_transforms(img, annot)

        if self.eval_mode:
            embedding = []
            att = []
            t_masks = []
            p_masks = []
            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                a = self.attention_masks[index][s]
                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))
                if self.use_gpg:
                    t = self.target_masks[index][s]
                    p = self.position_masks[index][s]
                    t_masks.append(t.unsqueeze(-1))
                    p_masks.append(p.unsqueeze(-1))

            tensor_embeddings = torch.cat(embedding, dim=-1)
            attention_mask = torch.cat(att, dim=-1)
            if self.use_gpg:
                target_mask = torch.cat(t_masks, dim=-1)
                position_mask = torch.cat(p_masks, dim=-1)
        else:
            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]
            if self.use_gpg:
                target_mask = self.target_masks[index][choice_sent]
                position_mask = self.position_masks[index][choice_sent]

        if self.eval_mode:
            meta = {
                'ref_id': this_ref_id,
                'img_id': this_img_id[0],
                'file_name': this_img['file_name'],
            }
            if self.use_gpg:
                return img, target, tensor_embeddings, attention_mask, target_mask, position_mask, meta
            return img, target, tensor_embeddings, attention_mask, meta
        if self.use_gpg:
            return img, target, tensor_embeddings, attention_mask, target_mask, position_mask
        return img, target, tensor_embeddings, attention_mask
