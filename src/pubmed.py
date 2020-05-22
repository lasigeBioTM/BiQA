# pubmed api interface
import json
import os
import time
import html
import requests
import spacy

from tqdm import tqdm

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_lg")

with open("params.json", "r") as f:
    params = json.load(f)


def get_doc_text(pmid, abstract_path="/pubmed_abstracts/"):
    """Retrieve text from PubMed files stored on disk

    :param pmid: PubMed ID to retrieve
    :type pmid: string
    :param abstract_path: directory where pubmed text files are stored as .txt files
    :type abstract_path: string
    :return: Title and abstract of article
    :rtype: string

    """
    if "http" in pmid:  # extract pmid
        pmid = pmid.split("/")[-1]

    if not os.path.isfile(abstract_path + pmid + ".txt"):
        # either the pmid is wrong or pubmed doesnt have an abstract in text format
        return ("", "")

    with open(abstract_path + pmid + ".txt") as f:
        text = f.readlines()

    if text[0].strip() == "":
        print("no text", text)
        return None
    return (text[0].strip(), " ".join(text[1:]).strip())


def get_pmids_for_query(query, n_docs, n_tokens=20, n_chars=500):
    """ Use PubMed entrez api to retrieve documents according to a query

    Query processing is performed on this function as it might differ from other
    retrieval engines.
    The system waits 0.1 seconds between each request to prevent from going over the 
    10 requests per second limit.

    :param query: Natural language query
    :type query: string
    :param n_docs: max number of documents to retrieve
    :type n_docs: int
    :param n_tokens: max number of token of the query
    :type n_tokens: int
    :param n_chars: max number of chars of the query (including URL)
    :type n_chars: int
    :return: list of PMIDs
    :rtype: list


    """
    # field=tiab&
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?api_key={}&db=pubmed&retmode=json&sort=relevance&retmax={}&term={}"
    query = html.unescape(query)
    doc = nlp(query)
    doc_tokens = [t for t in doc if not t.is_punct and not t.is_space and not t.is_stop]
    # doc_tokens = [t for t in doc if not t.is_punct and not t.is_space]
    # doc_tokens = [t for t in doc]
    doc_tokens = sorted(doc_tokens, key=lambda x: x.prob, reverse=False)
    # print([(t.text, t.prob) for t in doc_tokens])
    doc_tokens = list(dict.fromkeys([t.text.lower() for t in doc_tokens]))
    doc_tokens = doc_tokens[:n_tokens]
    # print(query, doc_tokens, file=sys.stderr)
    # q_articles[r["qid"]] = doc_tokens
    # print(doc_tokens)
    request_url = base_url.format(params["pubmed_api"], n_docs, "+OR+".join(doc_tokens))
    if len(request_url) > n_chars:
        print("long url! trimming to {}".format(n_chars))
        request_url = request_url[:n_chars]
    try:
        pubmed_results = requests.get(request_url)
    except:
        return []
    # print(request_url, pubmed_results.text)
    if pubmed_results.status_code != 200:
        print(pubmed_results.text)

    if "json" not in pubmed_results.headers.get("Content-Type"):
        print("Response content is not in JSON format.")
        print(pubmed_results.text)
        return []
    try:
        pubmed_results = pubmed_results.json()
    except json.decoder.JSONDecodeError:
        return []

    try:
        pmids = pubmed_results["esearchresult"]["idlist"]
    except KeyError:
        print("KEYERROR no IDs")
        pmids = []
    # print(request_url, len(pmids))
    time.sleep(0.1)
    return pmids


def get_pubmeds_for_questions(aueb_dic, n_docs=100, limit_queries=None):
    """Retrieve the pubmed articles associated with each question

    :param aueb_dic: AUEB format dictionary
    :type aueb_dic: dict
    :param n_docs: max number of docs to retrieve
    :type n_docs: int
    :param limit_queries: either a list or a number to limit queries
    :type limit_queries: int or list
    :return: PMIDs for each query, with score and rank
    :rtype: dict

    """
    nresults_count = []
    ret_docs = {}
    if isinstance(limit_queries, int):
        aueb_dic["queries"] = aueb_dic["queries"][:limit_queries]
    elif isinstance(limit_queries, list):
        # print("findin questions from list")
        # print(aueb_dic["queries"][:2])
        aueb_dic["queries"] = [
            r for r in aueb_dic["queries"] if r["query_id"] in limit_queries
        ]
        print(aueb_dic, limit_queries)
    for r in tqdm(aueb_dic["queries"]):
        pmids = get_pmids_for_query(r["query_text"], n_docs)
        qid = r["query_id"]
        if qid not in ret_docs:
            ret_docs[qid] = {}
        nresults_count.append(len(pmids))
        for i, pmid in enumerate(pmids):
            ret_docs[qid][pmid] = {"rank": i, "score": (len(pmids) - i) / len(pmids)}
    print("average n of results", sum(nresults_count) / len(nresults_count))
    return ret_docs
