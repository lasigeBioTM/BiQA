# galago interface
import json
import subprocess
import time
import unicodedata
import html
import spacy


# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_lg")


def write_galago_query_file(aueb_dic, n, limit_queries=None):
    """Generate query file to be processed by galago

    :param aueb_dic: AUEB format dict
    :type aueb_dict: dict
    
    """
    if isinstance(limit_queries, int):
        aueb_dic["queries"] = aueb_dic["queries"][:limit_queries]
    elif isinstance(limit_queries, list):
        # print("findin questions from list")
        # print(aueb_dic["queries"][:2])
        aueb_dic["queries"] = [
            r for r in aueb_dic["queries"] if r["query_id"] in limit_queries
        ]
        print(aueb_dic, limit_queries)
    print("writing galago queries")
    query_dic = {"queries": []}
    for r in aueb_dic["queries"]:
        query_text = r["query_text"].replace(".", " ")
        query_text = html.unescape(query_text)
        doc = nlp(query_text)
        # doc_tokens = [t for t in doc if t.is_alpha and not t.is_stop]
        doc_tokens = [
            t for t in doc if not t.is_punct and not t.is_space and not t.is_stop
        ]
        # doc_tokens = [t for t in doc]
        doc_tokens = sorted(doc_tokens, key=lambda x: x.prob, reverse=False)
        # print([(t.text, t.prob) for t in doc_tokens])
        doc_tokens = list(dict.fromkeys([t.text for t in doc_tokens]))
        doc_tokens = doc_tokens[:20]
        # print(query_text, doc_tokens)
        q = {
            "number": str(r["query_id"]),
            # "text": "#bm25({})".format(") #bm25(".join(doc_tokens)),
            "text": "#combine({})".format(" ".join(doc_tokens)),
        }
        query_dic["queries"].append(q)
    with open("galago_query.json", "w") as f:
        json.dump(query_dic, f)
    print("done")


def get_pmids_galago(aueb_dic, n=100, limit_queries=None):
    # write query file with all the queries
    write_galago_query_file(aueb_dic, n, limit_queries)
    ret_docs = {}
    galago_path = "galago/galago-3.14-bin/bin/galago"
    galago_args = [
        galago_path,
        "threaded-batch-search",
        "--threadCount=20",
        # "--verbose=true",
        "--caseFold=true",
        # "--mu=2000",
        # "--scorer=bm25",
        # "--lambda=0.2",
        "--index=/galago_pubmed_idx",
        "--requested={}".format(n),
        "galago_query.json",
    ]
    print(" ".join(galago_args))
    galago_process = subprocess.Popen(
        galago_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    print("running queries...")
    try:
        # timeout=600
        stdout, stderr = galago_process.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        galago_process.kill()
        stdout, stderr = galago_process.communicate()

    print("done, processing output")
    lines = stdout.splitlines()
    for l in lines:
        values = l.decode().split()
        if not values or values[-1] != "galago":
            # print(l.decode())
            continue

        try:
            rank = int(values[3])
            qid = values[0]
            pmid = values[2].split("/")[-1].split(".")[0]
            bm25 = float(values[4])
        except ValueError:
            print(values)
            continue
        if qid not in ret_docs:
            ret_docs[qid] = {}
        ret_docs[qid][pmid] = {"rank": rank, "bm25": bm25, "score": bm25}
    print("done, obtained results for {} qs".format(len(ret_docs)))
    return ret_docs
