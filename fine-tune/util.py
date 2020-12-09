import logging
import numpy as np
from collections import defaultdict

# BIO2BIEOS
def to_bioes(original_tags):
    def _change_prefix(original_tag, new_prefix):
        assert original_tag.find("-") > 0 and len(new_prefix) == 1
        chars = list(original_tag)
        chars[0] = new_prefix
        return "".join(chars)

    def _pop_replace_append(stack, bioes_sequence, new_prefix):
        tag = stack.pop()
        new_tag = _change_prefix(tag, new_prefix)
        bioes_sequence.append(new_tag)

    def _process_stack(stack, bioes_sequence):
        if len(stack) == 1:
            _pop_replace_append(stack, bioes_sequence, "S")
            # _pop_replace_append(stack, bioes_sequence, "U")
        else:
            recoded_stack = []
            _pop_replace_append(stack, recoded_stack, "E")
            # _pop_replace_append(stack, recoded_stack, "L")
            while len(stack) >= 2:
                _pop_replace_append(stack, recoded_stack, "I")
            _pop_replace_append(stack, recoded_stack, "B")
            recoded_stack.reverse()
            bioes_sequence.extend(recoded_stack)

    bioes_sequence = []
    stack = []

    for tag in original_tags:
        if tag == "O":
            if len(stack) == 0:
                bioes_sequence.append(tag)
            else:
                _process_stack(stack, bioes_sequence)
                bioes_sequence.append(tag)
        elif tag[0] == "I":
            if len(stack) == 0:
                stack.append(tag)
            else:
                this_type = tag[2:]
                prev_type = stack[-1][2:]
                if this_type == prev_type:
                    stack.append(tag)
                else:
                    _process_stack(stack, bioes_sequence)
                    stack.append(tag)
        elif tag[0] == "B":
            if len(stack) > 0:
                _process_stack(stack, bioes_sequence)
            stack.append(tag)
        else:
            raise ValueError("Invalid tag:", tag)

    if len(stack) > 0:
        _process_stack(stack, bioes_sequence)

    return bioes_sequence


# BIOES2BIO
def BIEOS2BIO(original_tags):
    def _change_prefix(original_tag, new_prefix):
        assert original_tag.find("-") > 0 and len(new_prefix) == 1
        chars = list(original_tag)
        chars[0] = new_prefix
        return "".join(chars)

    def _pop_replace_append(stack, bio_sequence, new_prefix):
        tag = stack.pop()
        new_tag = _change_prefix(tag, new_prefix)
        bio_sequence.append(new_tag)

    def _process_stack(stack, bio_sequence):
        if len(stack) == 1:
            _pop_replace_append(stack, bio_sequence, "B")
            # _pop_replace_append(stack, bioes_sequence, "U")
        else:
            recoded_stack = []
            _pop_replace_append(stack, recoded_stack, "I")
            # _pop_replace_append(stack, recoded_stack, "L")
            while len(stack) >= 2:
                _pop_replace_append(stack, recoded_stack, "I")
            _pop_replace_append(stack, recoded_stack, "B")
            recoded_stack.reverse()
            bio_sequence.extend(recoded_stack)

    bio_sequence = []
    stack = []

    for tag in original_tags:
        if tag == "O":
            if len(stack) == 0:
                bio_sequence.append(tag)
            else:
                _process_stack(stack, bio_sequence)
                bio_sequence.append(tag)
        elif tag[0] == "E":
            stack.append(tag)
        elif tag[0] == "I":
            stack.append(tag)
        elif tag[0] == "S":
            if len(stack) > 0:
                _process_stack(stack, bio_sequence)
            stack.append(tag)
        elif tag[0] == "B":
            if len(stack) > 0:
                _process_stack(stack, bio_sequence)
            stack.append(tag)
        else:
            raise ValueError("Invalid tag:", tag)

    if len(stack) > 0:
        _process_stack(stack, bio_sequence)

    return bio_sequence


# tags to BIEOS  输出的tags必须符合BIEOS。适合在CRF层后使用
def tag_to_spans(tags):
    spans = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        entity_name = tag[2:]
        if tag[0] == "U" or tag[0] == "S":
            spans.append((i, i, tag[2:]))
        elif tag[0] == "B":
            start = i
            while tag[0] != "L" and tag[0] != "E":
                i += 1
                if i > len(tags):
                    raise ValueError("Invalid tag sequence: %s" %
                                     (" ".join(tags)))
                tag = tags[i]
                if not (tag[0] == "I" or tag[0] == "L" or tag[0] == "E"):
                    raise ValueError("Invalid tag sequence: %s" %
                                     (" ".join(tags)))
                if tag[2:] != entity_name:
                    raise ValueError(
                        "Invalid entity name match: %s" % (" ".join(tags)))
            spans.append((start, i, tag[2:]))
        else:
            if tag != "O":
                raise ValueError("Invalid tag sequence: %s" % (" ".join(tags)))
        i += 1
    spans_text = "|".join(["%s,%s %s" % (s[0], s[1], s[2]) for s in spans])
    return spans_text


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename, mode='a', encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

# 针对 输出的  BIEOS/BMEOS 不规则的标签不召回
def compute_spans_bieos(tags):
    spans = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        entity_name = tag[2:]
        start = i
        if tag not in ['[SEP]', '[CLS]']:
            if tag[0] == 'S':
                spans.append((i, i, tag[2:]))
            elif tag[0] == 'B':
                if start != (len(tags) - 1):
                    while tags[start + 1][2:] == entity_name and tags[start + 1][0] != 'O' and tags[start + 1][
                            0] != 'S' and tags[start + 1][0] != 'B':
                        if tags[start][0] == 'E':
                            break
                        start += 1
                        if start == len(tags) - 1:
                            break
                    if tags[start][0] == 'E':
                        spans.append((i, start, entity_name))

        i += (start - i) + 1
    spans_text = "|".join(["%s,%s %s" % (s[0], s[1], s[2]) for s in spans])
    return spans_text

# 针对 输出的  BIO 不规则的标签不召回
def compute_spans_bio(tags):
    spans = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        entity_name = tag[2:]
        start = i
        if tag not in ['[SEP]', '[CLS]']:
            if tag[0] == 'B':
                if start != (len(tags) - 1):
                    while tags[start + 1][2:] == entity_name and tags[start + 1][0] == 'I':
                        start += 1
                        if start == len(tags) - 1:
                            break
                    spans.append((i, start, entity_name))
                else:
                    spans.append((i, start, entity_name))
        i += (start - i) + 1
    spans_text = "|".join(["%s,%s %s" % (s[0], s[1], s[2]) for s in spans])
    return spans_text


# 针对 预测输出不连续  BIEOS/BIO  不规则的标签也召回
def compute_nonspans(tags):
    spans = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        entity_name = tag[2:]
        start = i
        if tag not in ['[SEP]', '[CLS]']:
            if tag[0] == 'S':
                spans.append((i, i, tag[2:]))
            elif tag[0] != 'O':
                if start != (len(tags) - 1):
                    while tags[start + 1][2:] == entity_name and tags[start + 1][0] != 'O' and tags[start + 1][
                            0] != 'S' and tags[start + 1][0] != 'B':
                        if tags[start][0] == 'E':
                            break
                        start += 1
                        if start == len(tags) - 1:
                            break
                    spans.append((i, start, entity_name))
                else:
                    spans.append((i, start, entity_name))
        i += (start - i) + 1
    spans_text = "|".join(["%s,%s %s" % (s[0], s[1], s[2]) for s in spans])
    return spans_text


def test_spans():
    # test = ['B-PER', 'B-PER', 'I-PER', 'O', 'B-PER',
    #         'I-MISC', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER']
    # span = compute_spans_bio(test)
    # print(span)

    test = ['I-PER', 'O', 'B-MISC', 'I-MISC',
            'E-MISC', 'B-PER', 'E-PER', 'S-PER']
    print(compute_spans_bieos(test))
    # print(tag_to_spans(test))

    """
        计算F1样例 ['B-PER', 'E-PER', 'O', 'B-MISC', 'I-MISC', 'E-MISC', 'O'],  ['B-PER', 'E-PER', 'B-MISC', 'I-MISC', 'I-MISC', 'E-MISC', 'O']
    """

    y_true = [['B-SSS', 'E-SSS', 'O', 'S-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'E-PER', 'B-MISC', 'I-MISC', 'I-MISC',
                                                                           'S-MISC', 'O']]
    y_pred = [['B-ORG', 'E-ORG', 'O', 'B-MISC', 'I-MISC', 'E-MISC', 'O'], ['B-PER', 'E-PER', 'B-MISC', 'I-MISC', 'I-MISC',
                                                                           'S-MISC', 'O']]

    # gold_sentences = [compute_nonspans(i) for i in y_true]
    # pred_sentences = [compute_nonspans(i) for i in y_pred]
    # print(gold_sentences)
    # print(pred_sentences)

    # tags = ['B-MISC', 'E-MISC', 'S-MISC', 'B-PER', 'I-PER', 'I-PER', 'E-PER']
    # biotag = BIEOS2BIO(tags)
    # print(biotag)
    # print(to_bioes(biotag))

# test_spans()

def compute_accuracy(gold_corpus, pred_corpus):
    assert len(gold_corpus) == len(pred_corpus) and len(gold_corpus) > 0
    correct = 0
    for gold, pred in zip(gold_corpus, pred_corpus):
        if gold == pred:
            correct += 1
    return correct / len(gold_corpus)



def _compute_f1(TP, FP, FN):
    precision = float(TP) / float(TP + FP) if TP + FP > 0 else 0
    recall = float(TP) / float(TP + FN) if TP + FN > 0 else 0
    f1 = 2. * ((precision * recall) / (precision + recall)) if precision + recall > 0 else 0
    return precision, recall, f1


def compute_f1(gold_corpus, pred_corpus):
    assert len(gold_corpus) == len(pred_corpus)

    TP, FP, FN = defaultdict(int), defaultdict(int), defaultdict(int)
    for gold_sentence, pred_sentence in zip(gold_corpus, pred_corpus):
        gold_sentence = gold_sentence.strip().split("|") if len(gold_sentence.strip()) > 0 else []
        pred_sentence = pred_sentence.strip().split("|") if len(pred_sentence.strip()) > 0 else []

        for gold in gold_sentence:
            _, label = gold.split()
            if gold in pred_sentence:
                TP[label] += 1
            else:
                FN[label] += 1
        for pred in pred_sentence:
            _, label = pred.split()
            if pred not in gold_sentence:
                FP[label] += 1

    all_labels = set(TP.keys()) | set(FP.keys()) | set(FN.keys())
    metrics = {}

    macro_f1 = 0
    for label in all_labels:
        precision, recall, f1 = _compute_f1(TP[label], FP[label], FN[label])
        metrics["precision-%s" % label] = precision
        metrics["recall-%s" % label] = recall
        metrics["f1-measure-%s" % label] = f1
        macro_f1 += f1
    precision, recall, f1 = _compute_f1(sum(TP.values()), sum(FP.values()), sum(FN.values()))
    metrics["precision-overall"] = precision
    metrics["recall-overall"] = recall
    metrics["f1-measure-overall"] = f1
    metrics['micro-f1'] = f1
    metrics['macro-f1'] = macro_f1 / len(all_labels)

    return metrics


def compute_instance_f1(y_true, y_pred):
    metrics = {}
    TP, FP, FN = defaultdict(int), defaultdict(int), defaultdict(int)
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            if y_true[i][j] not in ['[SEP]', '[CLS]','[PAD]','O']:
                if y_true[i][j] == y_pred[i][j]:
                    TP[y_true[i][j][2:]] += 1
                else:
                    FN[y_true[i][j][2:]] += 1
        for k in range(len(y_pred[i])):
            if y_pred[i][k] not in ['[SEP]', '[CLS]','[PAD]','O']:
                if y_pred[i][k] != y_true[i][k]:
                    FP[y_pred[i][k][2:]] += 1

    all_labels = set(TP.keys()) | set(FP.keys()) | set(FN.keys())

    macro_f1 = 0
    for label in all_labels:
        precision, recall, f1 = _compute_f1(TP[label], FP[label], FN[label])
        metrics["precision-%s" % label] = precision
        metrics["recall-%s" % label] = recall
        metrics["f1-measure-%s" % label] = f1
        macro_f1 += f1
    precision, recall, f1 = _compute_f1(sum(TP.values()), sum(FP.values()), sum(FN.values()))
    metrics["precision-overall"] = precision
    metrics["recall-overall"] = recall
    metrics["f1-measure-overall"] = f1
    metrics['micro-f1'] = f1
    metrics['macro-f1'] = macro_f1 / len(all_labels)

    return metrics