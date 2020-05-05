import sys
import os
import time
import json
import pickle
from bs4 import BeautifulSoup
from stackapi import StackAPI
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from tqdm import tqdm
import logging

# SE answer retriever
request_query = True  # set to True to call SE API, False uses cached pickle
with open("params.json", "r") as f:
    params = json.load(f)


from qas import (
    a_cols,
    q_cols,
    numerical_cols,
    binary_cols,
    text_cols,
    show_output,
    print_stats,
    write_aueb_pickle,
    normalize_pmid,
)


def retrieve_questions(sitename):
    """Use stack exchange API to retrieve questions

    Can request from scratch (request_query=True) or return a previously cached request.
    Use cached request to repeat experiments wihtout overloading the API.

    :param sitename: Name of StackExchange community
    :type sitename: string
    :return: question objects
    :rtype: list 

    """
    # sitename = sitename.split("/")[-1]
    if request_query:
        SITE = StackAPI(sitename, key=params["se_key"])
        SITE.page_size = 50
        SITE.max_pages = 1000  # max qs should be page_size * max_pages
        questions = SITE.fetch(
            "questions", filter="!-*jbN-o8P3E5", sort="votes"
        )  # has q and a text
        with open("{}_questions_cache.json".format(sitename), "w") as f:
            json.dump(questions, f)
    else:
        with open("{}_questions_cache.json".format(sitename), "r") as f:
            questions = json.load(f)
    # answers["items"][0]['answer_id']
    print(
        "quota max",
        questions["quota_max"],
        "quota remaining",
        questions["quota_remaining"],
        "total",
        questions["total"],
        "page",
        questions["page"],
        file=sys.stderr,
    )
    print("retrieved {} questions".format(len(questions["items"])))
    return questions["items"]


def parse_questions(question_items, sitename, min_answer_count=1, min_q_score=1):
    """Parse retrieved question items and populate pandas dataframe tables 
    q_table and a_table

    Print a summary of the parsing process.

    :param question_items: question objects retrieved with *retrieve_questions*
    :type question_items: list
    :param sitename: Name of StackExchange community
    :type sitename: string
    :param min_answer_count: minimum number of answers of each question
    :type min_answer_count: int
    :param min_q_score: minimum number of votes
    :type min_q_score: int
    :return: questions table, answers table and q-a link table
    :rtype: tuple
    
    """
    url_string = "https://"
    pubmed_url_string = "ncbi.nlm.nih.gov/p"

    url_count = 0
    quote_count = 0
    both_count = 0
    low_score_skip = 0
    no_answer_skip = 0

    # The columns of the tables is defined in qas.py
    q_table = pd.DataFrame(columns=q_cols)
    a_table = pd.DataFrame(columns=a_cols)
    q_a = {}  # qid -> aid

    for q in tqdm(question_items):
        # skip questions according to filters
        if q["answer_count"] < min_answer_count:
            no_answer_skip += 1
            continue
        if q["score"] < min_q_score:
            low_score_skip += 1
            continue

        qid = str(q["question_id"])
        qbody = q["body"]
        qtitle = q["title"]
        q_a[qid] = []
        q_table = q_table.append(
            {
                "qid": qid,
                "score": q["score"],
                "q_title": qtitle,
                "q_body": qbody.replace("<img", "<a").replace("<hr>", ""),
            },
            ignore_index=True,
        )
        # accepted = q["is_accepted"]
        for a in q["answers"]:
            aid = a["answer_id"]
            atext = a["body"]
            accepted = a["is_accepted"]
            a_score = a["score"]
            all_links = []
            pubmed_links = []
            # if url_string in atext:
            parsed_answer = BeautifulSoup(atext, features="html.parser")
            # get all links of this answer assuming the a tag was used
            for link in parsed_answer.find_all("a"):
                all_links.append(link.get("href"))
                # normalize only direct mappings
                if pubmed_url_string in link.get("href"):
                    # print(link.get("href"), pubmed_url_string, file=sys.stderr)
                    # get only the normalized PMID links
                    try:
                        pmid = normalize_pmid(link.get("href"))
                    except:
                        print("not mapped", link.get("href"))
                        pmid = None
                        # normalize_pmid(link.get("href"))
                    if pmid:
                        pubmed_links.append(pmid)

            q_a[qid].append(aid)

            # each answer may have 0 or more links/PMIDs
            a_table = a_table.append(
                {
                    "aid": aid,
                    "qid": qid,
                    "accepted": accepted,
                    "score": a_score,
                    "nlinks": len(all_links),
                    "npubmeds": len(pubmed_links),
                    "hasquote": "<blockquote>" in atext,
                    "a_text": atext.replace("<img", "<a").replace("<hr>", ""),
                    "a_links": tuple(all_links),
                    "pubmed_links": tuple(pubmed_links),
                },
                ignore_index=True,
            )

            # count quotes
            if url_string in atext:
                url_count += 1
                if "<blockquote>" in atext:
                    both_count += 1
            if "<blockquote>" in atext:
                quote_count += 1

    # print summary
    print("total questions", len(q_table))
    print("total answers", len(a_table))
    print("total qa", len(q_a))
    print("with url", url_count)
    print("with quotes", quote_count)
    print("with both", both_count)
    print("low_score skip", low_score_skip)
    print("no answer skip", no_answer_skip)

    a_table = a_table.astype(
        dtype={
            "accepted": "bool",
            "score": "int64",
            "nlinks": "int64",
            "npubmeds": "int64",
            "hasquote": "bool",
        }
    )
    return q_table, a_table, q_a


def get_pubmed_titles(ids):
    """
    """
    titles = {}
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
    fetch_url += "db=pubmed&id={}&retmode=xml"
    # get titles
    titles_result = requests.get(fetch_url.format(",".join(ids)))
    parsed_answer = BeautifulSoup(titles_result.text, "lxml-xml")
    for p in parsed_answer.find_all("MedlineCitation"):
        titles[p.find("PMID").text] = p.find("ArticleTitle").text
    return titles


def main():
    # sitename = "se_biology/biology"
    # sitename = "se_medicalsciences/medicalsciences"
    sitename = sys.argv[1]
    outputdir = "se/" + sitename + "/"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    outputdir += params["version"]
    print("retrieving questions from ", sitename)
    q_items = retrieve_questions(sitename)
    # q_items = q_items[:50]
    print("parsing qas")
    q_table, a_table, q_a = parse_questions(q_items, sitename, min_q_score=-100)
    print("writing files")
    if params["write_data"]:
        with open("{}_qtable.pkl".format(outputdir), "wb") as f:
            pickle.dump(q_table, f)
        with open("{}_atable.pkl".format(outputdir), "wb") as f:
            pickle.dump(a_table, f)
        with open(outputdir + "_q_a.pkl", "wb") as f:
            pickle.dump(q_a, f)
    elif params["read_data"]:
        with open("{}_qtable.pkl".format(outputdir), "rb") as f:
            q_table = pickle.load(f)
        with open("{}_atable.pkl".format(outputdir), "rb") as f:
            a_table = pickle.load(f)
        with open(outputdir + "_q_a.pkl", "rb") as f:
            q_a = pickle.load(f)
        print(len(q_a))

    # use generic functions imported from qa.py
    print("analyze data")
    show_output(q_table, a_table, q_a, outputdir)
    print_stats(q_table, a_table)
    write_aueb_pickle(q_table, a_table, q_a, outputdir)


if __name__ == "__main__":
    main()
