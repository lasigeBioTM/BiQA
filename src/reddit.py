import re
import json
import pickle
import sys
import os
import tqdm
import requests
import praw
import pandas as pd
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

request_query = True  # set to True to call SE API, False uses cached pickle
with open("params.json", "r") as f:
    params = json.load(f)


def process_comment(comment, submission):
    if comment.parent_id != comment.link_id:
        # not top level
        return None, None
    a_text = comment.body
    # if "ncbi" in a_text:
    all_links = re.findall("(?P<url>https?://[^\s]+)", a_text)
    pubmed_links = re.findall("(?P<url>https?://www.ncbi.nlm.nih.gov/p[^\s]+)", a_text)
    pmids = []
    for url in pubmed_links:
        try:
            pmid = normalize_pmid(url)
        except:
            print("not mapped", url)
            pmid = None
        if pmid:
            pmids.append(pmid)
    if len(all_links) > 0:
        # all_links = all_links.group("url")
        pubmed_qa_object = {
            "q_title": submission.title,
            "q_body": submission.selftext,
            "a_body": a_text,
        }
        

        # print(all_links)
        # else:
        #    all_links = []
        # q_a[submission.id].append(comment.id)
        a_object = {
            "aid": comment.id,
            "qid": submission.id,
            "accepted": False,
            "score": comment.score,
            "nlinks": len(all_links),
            "npubmeds": len(pmids),
            "hasquote": ">" in a_text,
            "a_text": a_text.replace("<img", "<a").replace("<hr>", ""),
            "a_links": all_links,
            "pubmed_links": tuple(pmids),  # direct pm links
        }

    else:
        return None, None
    return pubmed_qa_object, a_object


def get_reddit_questions_pushshift(sitename, min_answer_count=1, min_q_score=1):
    page_size = 1000
    base_url = "https://api.pushshift.io/reddit/search/submission/?size={}&sort_type=created_utc&subreddit={}"
    base_url = base_url.format(str(page_size), sitename)
    reddit = praw.Reddit(params["toolname"])

    pubmed_url_string = "https://www.ncbi.nlm.nih.gov/p"
    pubmed_qa = []
    q_table = pd.DataFrame(columns=q_cols)
    a_table = pd.DataFrame(columns=a_cols)
    q_a = {}

    question_items = []
    last_date = 0
    total_retrieved = 0
    iteration = 1
    while total_retrieved < 50000:
        if total_retrieved > 0:
            url = base_url + "&before={}".format(str(last_date))
        else:
            url = base_url
        print(iteration, url)
        result = requests.get(url)
        reddit_posts = result.json()
        if len(reddit_posts["data"]) == 0:
            break
        #last_score = reddit_posts["data"][-1]["score"]

        last_date = reddit_posts["data"][-1]["created_utc"]
        total_retrieved += len(reddit_posts["data"])
        print(
            reddit_posts["data"][0]["title"],
            last_date,
            total_retrieved,
            len(question_items),
        )

        for post in reddit_posts["data"]:
            new_question = {
                "score": post["score"],
                "question_id": post["id"],
                "body": post.get("selftext", ""),
                "title": post["title"],
            }

            if "?" in new_question["title"] or "?" in new_question["body"]:
                # retrieve submission using PRAW to get answers
                submission = reddit.submission(id=new_question["question_id"])
                
                q_a[submission.id] = []
                q_table = q_table.append(
                    {
                        "qid": submission.id,
                        "score": submission.score,
                        "q_title": submission.title,
                        "q_body": submission.selftext.replace("<img", "<a").replace(
                            "<hr>", ""
                        ),
                    },
                    ignore_index=True,
                )

                # get answers

                # print(submission.num_comments)
                submission.comments.replace_more(limit=0)
                for comment in submission.comments:

                    pubmed_qa_object, a_object = process_comment(comment, submission)

                    if pubmed_qa_object is not None:
                        pubmed_qa.append(pubmed_qa_object)
                        a_table = a_table.append(a_object, ignore_index=True)
                        q_a[submission.id].append(comment.id)
                question_items.append(new_question)
        iteration += 1
    return q_table, a_table, q_a


def get_reddit_questions(sitename, min_answer_count=1, min_q_score=1):
    """Use reddit API to retrieve questions

    Can request from scratch (request_query=True) or return a previously cached request.

    :param sitename: Name of reddit community
    :type sitename: string
    :param min_answer_count: minimum number of answers of each question
    :type min_answer_count: int
    :param min_q_score: minimum number of votes
    :type min_q_score: int
    :return: question objects
    :rtype: list 

    """
    reddit = praw.Reddit(params["toolname"])
    pubmed_url_string = "https://www.ncbi.nlm.nih.gov/p"
    pubmed_qa = []
    q_table = pd.DataFrame(columns=q_cols)
    a_table = pd.DataFrame(columns=a_cols)
    q_a = {}
    for submission in tqdm.tqdm(reddit.subreddit(sitename).top(limit=10000)):
        if submission.num_comments < min_answer_count:
            continue
        if submission.score < min_q_score:
            continue
        if "?" in submission.title or "?" in submission.selftext:
            # print("######################## QUESTION ##################")
            # print(submission.title, submission.selftext)

            q_a[submission.id] = []
            q_table = q_table.append(
                {
                    "qid": submission.id,
                    "score": submission.score,
                    "q_title": submission.title,
                    "q_body": submission.selftext.replace("<img", "<a").replace(
                        "<hr>", ""
                    ),
                },
                ignore_index=True,
            )

            submission.comments.replace_more(limit=None)
            comments = submission.comments.list()
            # print("########################### ANSWERS ##############")
            for comment in comments:
                if comment.parent_id != comment.link_id:
                    # not top level
                    continue
                a_text = comment.body
                # if "ncbi" in a_text:
                all_links = re.findall("(?P<url>https?://[^\s]+)", a_text)
                pubmed_links = re.findall(
                    "(?P<url>https?://www.ncbi.nlm.nih.gov/p[^\s]+)", a_text
                )
                pmids = []
                for url in pubmed_links:
                    try:
                        pmid = normalize_pmid(url)
                    except:
                        print("not mapped", url)
                        pmid = None
                    if pmid:
                        pmids.append(pmid)
                if len(all_links) > 0:
                    # all_links = all_links.group("url")
                    pubmed_qa.append(
                        {
                            "q_title": submission.title,
                            "q_body": submission.selftext,
                            "a_body": a_text,
                        }
                    )
                    # print(all_links)
                    # else:
                    #    all_links = []
                    q_a[submission.id].append(comment.id)
                    a_table = a_table.append(
                        {
                            "aid": comment.id,
                            "qid": submission.id,
                            "accepted": False,
                            "score": comment.score,
                            "nlinks": len(all_links),
                            "npubmeds": len(pmids),
                            "hasquote": ">" in a_text,
                            "a_text": a_text.replace("<img", "<a").replace("<hr>", ""),
                            "a_links": all_links,
                            "pubmed_links": tuple(pmids),
                        },
                        ignore_index=True,
                    )
                # print(comment.body)
                # print("##########################")
            # print()

            # for x in pubmed_qa:
            #    print("### question title ###")
            #    print(x["q_title"])
            #    print("### question body ###")
            #    print(x["q_body"])
            #    print("### answer ###")
            #    print(x["a_body"])
            #   print()
            #   print()
    print("TOTAL QA PAIRS:", len(pubmed_qa))
    return q_table, a_table, q_a


def main():
    # sitename = "nutrition"
    sitename = sys.argv[1]
    outputdir = "reddit/" + sitename + "/"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    # q_items = retrieve_questions(sitename)
    if request_query:
        # q_table, a_table, q_a = get_reddit_questions(sitename)
        q_table, a_table, q_a = get_reddit_questions_pushshift(sitename)
        sitename = "reddit/{}/{}".format(sitename, params["version"])
        with open("{}_qtable.pkl".format(sitename), "wb") as f:
            pickle.dump(q_table, f)
        with open("{}_atable.pkl".format(sitename), "wb") as f:
            pickle.dump(a_table, f)
        with open(sitename + "_q_a.pkl", "wb") as f:
            pickle.dump(q_a, f)
    else:
        sitename = "reddit/{}/{}".format(sitename, params["version"])
        with open("{}_qtable.pkl".format(sitename), "rb") as f:
            q_table = pickle.load(f)
        with open("{}_atable.pkl".format(sitename), "rb") as f:
            a_table = pickle.load(f)
        with open(sitename + "_q_a.pkl", "rb") as f:
            q_a = pickle.load(f)
    show_output(q_table, a_table, q_a, sitename)
    print_stats(q_table, a_table)
    write_aueb_pickle(q_table, a_table, q_a, sitename)
    # generate_plots(answer_data, sitename)
    # get_pubmeds_for_questions(q_table, a_table, 20)


if __name__ == "__main__":
    main()
