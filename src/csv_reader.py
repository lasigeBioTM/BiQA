import sys
import csv
import pickle
import argparse
import os
import atexit
from collections import Counter
from qas import normalize_pmid
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="bs4")
from stackapi import StackAPI
import json
import pprint
import tqdm

import praw


from pubmed import get_doc_text
from qas import calculate_semantic_similarity

"""
This script is used to re-process a corpus in CSV format that has been manually
annotated.
No manual annotations were used for v1 of the corpus because we saw little improvement
when considering the effort necessary.
A cache of previously retrieved posts is necessary.
This cache is generated with the stackexchange_questions.py or reddit.py scripts.

"""

with open("params.json", "r") as f:
    params = json.load(f)


def get_se_question(qid):
    """Retrieve a specific question using stackexchange cache
    
    :param qid: Question ID
    :type qid: string
    :param se_cache: cached SE search results
    :type se_cache: dict
    :return: question item or None if not found
    :rtype: dict
    """
    for q in cache["items"]:
        # print(qid, q["question_id"])
        if str(q["question_id"]) == str(qid):
            # q_cache[qid] = q
            return q


def get_answer(aid):
    for q in cache["items"]:
        # print(q.keys())
        if "answers" in q:
            for a in q.get("answers"):
                if str(a["answer_id"]) == aid:
                    # cache[aid] = a
                    return a


def get_reddit_post(qid):
    if qid in cache and "score" in cache[qid]:
        return cache[qid]
    else:
        submission = reddit.submission(id=qid)
        q_object = {
            "body": submission.selftext.replace("<img", "<a").replace("<hr>", ""),
            "score": submission.score
        }
        cache[qid] = q_object
        return q_object


def get_reddit_comment(aid):
    if aid in cache:
        return cache[aid]
    else:
        submission = reddit.comment(id=aid)
        q_object = {"body": submission.body.replace("<img", "<a").replace("<hr>", "")}
        cache[aid] = q_object
    return q_object


def get_column_indexes(filename):
    """Get CSV corpus file column names and indexes according to filename

    :param filename: filename of CSV corpus
    :type filename: string
    :return: Dictionary of the column names of CSV file
    :rtype: dict
    """
    indexes = {
        "link_index": 8,
        "qid_index": 0,
        "aid_index": 1,
        "qtext_index": 6,
        "score_index": 3,
        "atext_index": 7,
    }

    # if the file has been annotated, the column numbers are different
    if "alinks" in filename or "annotated" in filename:
        indexes = {
            "link_index": 11,
            "qid_index": 3,
            "aid_index": 4,
            "qtext_index": 9,
            "score_index": 6,
            "atext_index": 10,
        }
    return indexes


def generate_q_text(row, use_title, use_body, use_answer, filename, idx):
    """Generate query text according to question title and/or body
    
    :param row: Row object from CSV corpus file
    :type row:
    :param use_title: Use question title text for query
    :type use_title: boolean
    :param use_body: Use question body text for query
    :type use_body: boolean
    :param use_body: Use answer text for query
    :type use_body: boolean
    :param filename: Filename of corpus CSV file
    :type filename: string
    :param idx: Dictionary of the column names of CSV file
    :type idx: dict
    :return: query text
    :rtype: string

    """
    if use_title:
        qtext = row[idx["qtext_index"]].strip()
    else:
        qtext = ""
    if use_body:  # retrieve body text from cache
        if "reddit" not in filename:
            q_object = get_se_question(row[idx["qid_index"]])
        else:
            q_object = get_reddit_post(row[idx["qid_index"]])
        if not q_object:
            # print("no body", row[idx["qid_index"]])
            return qtext
        body = q_object.get("body", "")
        if body:  # parse HTML to remove tags
            soup = BeautifulSoup(body, "html.parser")
            body = soup.get_text()
            qtext += " " + body.strip()  # also tags

    # IR idea: use answer text instead of question text to retrieve documents
    if use_answer:
        aid = row[idx["aid_index"]]
        if "reddit" not in filename:
            a_object = get_answer(aid)
        else:
            a_object = get_reddit_comment(aid)
        if a_object:
            body = a_object.get("body", "")
            if body:
                soup = BeautifulSoup(body, "html.parser")
                body = soup.get_text()
                # print("adding answer text", row[idx["qid_index"]])
                qtext += " " + body.strip()

    return qtext


def get_question_score(row, filename, idx):
    if "reddit" not in filename:
        q_object = get_se_question(row[idx["qid_index"]])
    else:
        q_object = get_reddit_post(row[idx["qid_index"]])
    if not q_object:
        # print("no body", row[idx["qid_index"]])
        return 0
    return q_object["score"]



def process_csv_file(
    origin_file, dest_name, min_a_score, min_a_count, use_title, use_body, use_answer, slowmode=True
):
    """write CSV corpus with filters applied based on another CSV file with only 
    links mapped to pubmed.

    Prints counts according to *print_counters*

    :param origin_file: input CSV corpus file path
    :type origin_file: string
    :param dest_name: output CSV corpus file path
    :type dest_name: string
    :param min_a_score: Answer score cutoff value
    :type min_a_score: int
    :param min_a_count: Minimum number of docs associated with a question
    :type min_a_count: int
    :param use_title: Use question title text for query
    :type use_title: boolean
    :param use_body: Use question body text for query
    :type use_body: boolean
    :param use_answer: Use answer text for query
    :type use_answer: boolean

    """

    idx = get_column_indexes(origin_file)

    pkl_dest_name = dest_name + ".pkl"
    csv_dest_name = dest_name + ".csv"
    docs_f = open(csv_dest_name, "w")

    # CSV corpus
    corpus_file = csv.writer(docs_f)
    corpus_file.writerow(
        ["question_id", "answer_id", "question_text", "question_score", "pmid", "pmtitle"]
    )
    csv_lines = []

    # Aueb format pickle
    final_dic = {"queries": []}
    qs = {}

    # counters
    counters = {}
    counters["a_pubmed_counts"] = {}
    counters["q_pubmed_counts"] = {}
    counters["no_pubmed_count"] = 0
    counters["qs_with_pubmed"] = set()
    counters["below_score_count"] = 0
    counters["a_scores"] = {}
    counters["all_qs"] = set()
    num_lines = sum(1 for line in open(origin_file, "r"))
    with open(origin_file, "r") as f:
        csvreader = csv.reader(f)
        next(f)
        # lines = [line for line in f]
        for r in tqdm.tqdm(csvreader, total=num_lines):
            # skip answers without enough links
            if len(r) < idx["link_index"]:
                counters["no_link_count"] += 1
                continue
            if int(r[idx["score_index"]]) < min_a_score:
                counters["below_score_count"] += 1
                continue

            qid = r[idx["qid_index"]]
            aid = r[idx["aid_index"]]
            counters["all_qs"].add(qid)
            if qid not in counters["q_pubmed_counts"]:
                counters["q_pubmed_counts"][qid] = 0
            qtext = generate_q_text(
                r, use_title, use_body, use_answer, origin_file, idx
            )

            qscore = get_question_score(r, origin_file, idx)
            #print(qscore)

            if qid not in qs:
                # AUEB system format
                qs[qid] = {
                    "query_id": qid,
                    "query_text": qtext,
                    "relevant_documents": set(),
                    "num_rel": 0,
                    "retrieved_documents": {},
                    "num_ret": 0,
                    "num_rel_ret": 0,
                }
            links = r[idx["link_index"]].split(",")
            # print(qid, links)
            # a_pmids = 0
            if aid not in counters["a_pubmed_counts"]:
                counters["a_pubmed_counts"][aid] = 0
            if aid not in counters["a_scores"]:
                counters["a_scores"][aid] = int(r[idx["score_index"]])

            # normalize links to pubmed
            for l in links:
                l = l.lower()
                if (
                    "/pubmed/" in l
                    or "/pmc/articles/" in l
                    or "doi.org" in l
                    or "researchgate" in l
                    or "sciencedirect" in l
                    or "accid=" in l
                    or "pmid=" in l
                ):  # use only pubmed and pmc or doi.org
                    clean_link = l.split("(")[-1].split(")")[0]
                    if len(clean_link) < 5:  # DOIs can have parenthesis
                        clean_link = l
                    # print(l, clean_link)
                    doc_id = normalize_pmid(clean_link, revisit_missing=slowmode)
                    # print(l, doc_id)
                    if doc_id is None:
                        continue
                    # print(doc_id)
                    # print("doc_id", doc_id, qs[qid]["relevant_documents"])
                    if doc_id not in qs[qid]["relevant_documents"]:
                        # a_pmids += 1
                        counters["a_pubmed_counts"][aid] += 1
                        counters["q_pubmed_counts"][qid] += 1
                        qs[qid]["relevant_documents"].add(doc_id)
                        doc_text = get_doc_text(doc_id)
                        csv_lines.append(
                            [
                                qid,
                                aid,
                                qtext.replace("\n", " "),
                                #r[idx["score_index"]],
                                qscore,
                                doc_id,
                                doc_text[0],
                            ]
                        )

            # do not keep counting score and links of this answer if we did not
            # get a normalized pubmed link
            if counters["a_pubmed_counts"][aid] == 0:
                del counters["a_pubmed_counts"][aid]
                del counters["a_scores"][aid]

    # calculate rel and ret doc totals
    excluded_qids = []
    qs_keys = list(qs.keys())
    for q in qs_keys:
        qs[q]["num_rel"] = len(qs[q]["relevant_documents"])
        if qs[q]["num_rel"] < min_a_count:
            del qs[q]
            del counters["q_pubmed_counts"][q]
            # del counters["a_scores"][aid]
            excluded_qids.append(q)
            continue
        # print(qs[q]["num_rel"], min_a_count, qs[q]["relevant_documents"])
        qs[q]["relevant_documents"] = list(qs[q]["relevant_documents"])

        final_dic["queries"].append(qs[q])
        if qs[q]["num_rel"] > 0:
            counters["qs_with_pubmed"].add(q)

            # only PMIDs are writen to CSV corpus
        # TODO: only write if > a_min_count
    for l in csv_lines:
        if l[0] not in excluded_qids:
            corpus_file.writerow(l)
    docs_f.close()

    with open(pkl_dest_name, "wb") as f:
        pickle.dump(final_dic, f)

    print_counters(counters)
    return csv_lines


def print_counters(counters):
    """ Print counts obtained by parsing a CSV corpus

    :param counters: Each key is a specific count related to the corpus
    :type counters: dict

    """
    pp = pprint.PrettyPrinter(indent=2)
    # print stats
    print("all qs", len(counters["all_qs"]))
    print("qs_with_pubmed", len(counters["qs_with_pubmed"]))
    print(
        "as_with_pubmed",
        len(
            [
                a
                for a in counters["a_pubmed_counts"]
                if counters["a_pubmed_counts"][a] > 0
            ]
        ),
        len(counters["a_pubmed_counts"]),
    )

    print("below_score_count", counters["below_score_count"])
    print("no pubmed count", counters["no_pubmed_count"])
    print("total Q-pmid pairs", sum(counters["a_pubmed_counts"].values()))
    print(
        "avg pubmeds per A",
        sum(counters["a_pubmed_counts"].values()) / len(counters["a_pubmed_counts"]),
    )
    
    #print("#total links", Counter(a_pubmed_counts.values()))
    """print()
    print("#pubmed links count table")
    counts = Counter(counters["a_pubmed_counts"].values())
    for i in range(min(counts.keys()), max(counts.keys()) + 1):
        print(i, counts.get(i, 0))
    print()
    print("scores dist count table")
    counts = Counter(counters["a_scores"].values())
    for i in range(min(counts.keys()), max(counts.keys()) + 1):
        print(i, counts.get(i, 0))
    """
    print(
        "average A score",
        sum(counters["a_scores"].values()) / len(counters["a_scores"].values()),
    )


def main():
    global cache
    global cache_file
    global reddit
    parser = argparse.ArgumentParser(description="read csv corpus, write tables.")
    parser.add_argument("file", type=str, help="csv file to be processed")
    parser.add_argument("--cache", type=str, default=None, help="cache file to be used")
    # parser.add_argument("--sitename", type=str, help="sitename")
    parser.add_argument(
        "--min_a_score", type=int, default=-100, help="minimum answer score"
    )
    parser.add_argument(
        "--min_a_count", type=int, default=1, help="minimum number of links"
    )
    parser.add_argument("--body_text", action="store_true", help="use body text")
    parser.add_argument("--title_text", action="store_true", help="use title text")
    parser.add_argument("--answer_text", action="store_true", help="use answer text")

    args = parser.parse_args()

    if "reddit" in args.file:
        reddit = praw.Reddit(params["toolname"])
        cache_file = "reddit_cache.pkl"
        if os.path.isfile(cache_file):
            with open(cache_file, "r") as f:
                cache = json.load(f)
        else:
            cache = {}
    else:
        # use a cache of retrieved posts texts
        cache_file = args.cache
        if os.path.isfile(cache_file):
            # logging.info("loading cache...")
            with open(cache_file, "r") as f:
                cache = json.load(f)
        # logging.info("loaded cache dictionary with %s entries", str(len(pm_cache)))
        else:
            print("cache file not found")
            sys.exit()

    # generate filename according to options
    dest_name = args.file[:-4] + "_ascore{}_acount{}".format(
        args.min_a_score, args.min_a_count
    )
    if args.title_text:
        dest_name += "_title"
    if args.body_text:
        dest_name += "_body"
    if args.answer_text:
        dest_name += "_answer"

    print("processing ", args.file)
    csv_lines = process_csv_file(
        args.file,
        dest_name,
        args.min_a_score,
        args.min_a_count,
        args.title_text,
        args.body_text,
        args.answer_text,
        slowmode=False
    )

    if "reddit" in args.file:
        with open(cache_file, "w") as f:
            json.dump(cache, f)
    #print(csv_lines)
    calculate_semantic_similarity(csv_lines)

if __name__ == "__main__":
    main()
