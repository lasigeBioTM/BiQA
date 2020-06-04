import sys
import pickle
import os
import csv
import atexit
import requests
import urllib.parse
import json
import re
from random import sample   
from tqdm import tqdm
import spacy
import numpy as np

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_vectors_web_lg")

"""
Helper functions for QA post retrieval and processing.
"""

with open("params.json", "r") as f:
    params = json.load(f)

# use cache to avoid frequent calls to ID converter API
cache_file = "pmid_maping.pickle"
# store string-> pubmed ID
if os.path.isfile(cache_file):
    # logging.info("loading cache...")
    pm_cache = pickle.load(open(cache_file, "rb"))
    if "None" not in pm_cache:
        pm_cache["None"] = set()
    loadedcache = True
    # logging.info("loaded cache dictionary with %s entries", str(len(pm_cache)))
else:
    pm_cache = {}
    pm_cache["None"] = set()
    loadedcache = False
    print("new pmid cache")


def exit_handler():
    print("Saving cache dictionary...!")
    pickle.dump(pm_cache, open(cache_file, "wb"))


atexit.register(exit_handler)

# Question and Answer tables
a_cols = [
    "aid",
    "qid",
    "accepted",
    "score",
    "nlinks",
    "npubmeds",
    "hasquote",
    "a_text",
    "a_links",
]

q_cols = ["qid", "score", "q_title", "q_body"]

numerical_cols = ["score", "nlinks", "npubmeds"]
binary_cols = ["accepted", "hasquote"]
text_cols = ["q_title", "q_body", "a_text"]


def calculate_semantic_similarity(csvlines):
    """
    csv columns: qid, aid, qtext, score, docid, doctext
    """
    sim_values = []
    random_sim_values = []
    for i, line in enumerate(tqdm(csvlines)):
        #print("this line", line)
        if len(line) <5:
            print("line:", line)
            break
        if line[5].strip() == "":
            continue
        doc1 = nlp(line[2])
        doc2 = nlp(line[5])
        if not doc1.vector_norm or not doc2.vector_norm:
            continue
        sim_values.append(doc1.similarity(doc2))
        
        #print("random lines", random_lines)
        random_compare = 1
        random_lines = sample(csvlines, random_compare)
        for irandom in range(random_compare):
            while True:
                if len(random_lines[irandom]) < 5  or \
                    random_lines[irandom][0] == line[0] or \
                    random_lines[irandom][5].strip() == "":
                    #print(len(random_lines[irandom]), random_lines[irandom][0], line[0]) #, random_lines[irandom][5])
                    #print("not using this random", irandom, random_lines[irandom])
                    random_lines[irandom] = sample(csvlines,1)[0]
                else:
                    break
        for r in random_lines:
            doc3 = nlp(r[5])
            if not doc3.vector_norm:
                continue
            random_sim_values.append(doc1.similarity(doc3))
        if i % 500 == 0:
            print(line[2], line[5])
            print(sum(sim_values)/len(sim_values))
            print(sum(random_sim_values)/len(random_sim_values))
    print()
    print("qa size", len(sim_values))
    print("qa average:", np.mean(sim_values))
    print("qa std", np.std(sim_values))
    print()
    print("qrandom size", len(random_sim_values))
    print("qrandom average:", np.mean(random_sim_values))
    print("qrandom std", np.std(random_sim_values))
    with open("sim_scores.csv", 'w') as scoresfile:
        scoresfile.write("\n".join([str(s) for s in sim_values]) + "\n")
    with open("random_sim_scores.csv", 'w') as scoresfile:
        scoresfile.write("\n".join([str(s) for s in random_sim_values]) + "\n")


def write_aueb_pickle(q_table, a_table, q_a, sitename):
    """Write pickle file in the format that is excepted from the AUEB team for bioasq

    Writes to sitename + aueb.pickle
    
    :param q_table: question pandas dataframe table
    :type q_table: pandas.DataFrame
    :param a_table: answer pandas dataframe table
    :type a_table: pandas.DataFrame
    :param q_a: q-a mapping table
    :type q_a: pandas.DataFrame
    :param sitename: name of community
    :type sitename: string
    """
    final_dic = {"queries": []}
    for q in q_a:
        # consider only at least one pubmed in the answers:
        if not any(
            [a["npubmeds"] > 0 for _, a in a_table[a_table["qid"] == q].iterrows()]
        ):
            continue
        rel_docs = []
        num_rel = 0
        # merge relevant docs from every a associated with a q
        for i, r in a_table[a_table["qid"] == q].iterrows():
            rel_docs += r["pubmed_links"]
            num_rel += r["npubmeds"]
        # either use the q_title or q_body or both
        query_text = (
            str(q_table.loc[q_table["qid"] == q]["q_title"].values[0])
            # + " "
            # + str(q_table.loc[q_table["qid"] == q]["q_body"].values[0])
        )
        new_q = {
            "query_id": q,
            "query_text": query_text,
            "relevant_documents": rel_docs[:],
            "num_rel": num_rel,
            "retrieved_documents": {},
            "num_ret": 0,
            "num_rel_ret": 0,
        }
        final_dic["queries"].append(new_q)
        # file name expected by aueb system
    with open(sitename + ".aueb.pkl", "wb") as f:
        pickle.dump(final_dic, f)


def show_output(q_table, a_table, q_a, sitename):
    """Write an HTML report file and CSV corpus

    Includes all links (not just PMIDs)

    :param q_table: question pandas dataframe table
    :type q_table: pandas.DataFrame
    :param a_table: answer pandas dataframe table
    :type a_table: pandas.DataFrame
    :param q_a: q-a mapping table
    :type q_a: pandas.DataFrame
    :param sitename: name of community
    :type sitename: string

    """

    html_output_file = open("{}.html".format(sitename), "w")
    docs_f = open("{}_qdocs.csv".format(sitename), "w")
    docs_file = csv.writer(docs_f)
    docs_file.writerow(
        [
            "qid",
            "aid",
            "accepted",
            "score",
            "#links",
            "#pmlinks",
            "q_title",
            "a_text",
            "links",
        ]
    )
    quotes_file =  csv.writer(open("{}_quotes.csv".format(sitename), "w"))
    quotes_file.writerow(
        [
            "qid",
            "aid",
            "accepted",
            "score",
            "q_title",
            "a_text",
            "links",
        ]
    )
    print(
        """<!DOCTYPE html>
    <html>
    <head>
    <style>
    table, th, td {
      border: 1px solid black;
      border-collapse: collapse;
    }
    </style>
    </head>
    <body>""",
        file=html_output_file,
    )
    for q in q_a:
        print(q_table.loc[q_table["qid"] == q]["q_title"].values, file=html_output_file)
        print(
            """<table style="width:100%">
          <tr>
            <th>""",
            file=html_output_file,
        )
        print("</th><th>".join(a_table.columns.values), file=html_output_file)
        print("</tr>", file=html_output_file)
        for i, r in a_table[a_table["qid"] == q].iterrows():
            """qid = int(r[-3])
            if qid not in questions:
                print("question not found", qid, r, file=sys.stderr)
                # pass
            else:
                r[-3] = (
                    questions[qid]["title"]
                    + "<br>"
                    + questions[qid]["body"].replace("<img", "<a").replace("<hr>", "")
                )"""
            # print to TSV file: "aid", "accepted", "score", "#links", "qtitle", "links"
            # print(q_table.loc[q_table["qid"] == q]["q_title"])
            for link in r["a_links"]:
                if (
                    "en.wikipedia" in link
                    or "reddit.com" in link
                    or "stackexchange.com" in link
                ):
                    # these are not going to be converted to pmids
                    continue
                docs_file.writerow(
                    [
                        str(q),
                        str(r["aid"]),
                        str(r["accepted"]),
                        str(r["score"]),
                        str(r["nlinks"]),
                        str(r["npubmeds"]),
                        str(q_table.loc[q_table["qid"] == q]["q_title"].values[0]),
                        str(r["a_text"][:50]).replace("\n", " "),
                        link,
                    ]
                )
                if r["hasquote"]:
                    quotes_file.writerow(
                    [
                        str(q),
                        str(r["aid"]),
                        str(r["accepted"]),
                        str(r["score"]),
                        str(q_table.loc[q_table["qid"] == q]["q_title"].values[0]),
                        str(r["a_text"]).replace("\n", " "),
                        link,
                    ]
                )
            # print to html page
            print("<tr><td>", file=html_output_file)
            print("</td><td>".join([str(v) for v in r]), file=html_output_file)
            print("</td></tr>", file=html_output_file)

        print("</table><br>", file=html_output_file)
    print("</body></html>", file=html_output_file)

    docs_f.close()
    html_output_file.close()


def print_stats(q_table, a_table):
    """Calculate and print stats of q and a tables
    
    :param q_table: question pandas dataframe table
    :type q_table: pandas.DataFrame
    :param a_table: answer pandas dataframe table
    :type a_table: pandas.DataFrame

    """
    numerical_values = a_table.loc[:, numerical_cols + binary_cols]
    print(
        "all QAs:",
        len(numerical_values),
        "qs:",
        len(set(a_table.loc[a_table["nlinks"] > 0]["qid"])),
        "as:",
        file=sys.stderr,
    )
    # print(numerical_values, file=sys.stderr)
    print("mean:", file=sys.stderr)
    # print(a_data.loc[:, numerical_cols].mean(axis=0, skipna=True), file=sys.stderr)
    print(
        numerical_values.agg(["min", "max", "mean", "median", "std", "var"]),
        file=sys.stderr,
    )
    print("correlation", file=sys.stderr)
    print(numerical_values.corr(method="spearman"), file=sys.stderr)

    print(
        "w/ links only",
        "qs:",
        len(set(a_table.loc[a_table["nlinks"] > 0]["qid"])),
        "as:",
        len(numerical_values.loc[numerical_values["nlinks"] > 0]),
        file=sys.stderr,
    )
    print(
        numerical_values.loc[numerical_values["nlinks"] > 0].agg(
            ["min", "max", "mean", "median", "std", "var"]
        ),
        file=sys.stderr,
    )
    print(
        numerical_values.loc[numerical_values["nlinks"] > 0].corr(method="spearman"),
        file=sys.stderr,
    )

    print(
        "pm only",
        "qs:",
        len(set(a_table.loc[a_table["npubmeds"] > 0]["qid"])),
        "as:",
        len(numerical_values.loc[numerical_values["npubmeds"] > 0]),
        file=sys.stderr,
    )
    print(
        numerical_values.loc[numerical_values["npubmeds"] > 0].agg(
            ["min", "max", "mean", "median", "std", "var"]
        ),
        file=sys.stderr,
    )
    print(
        numerical_values.loc[numerical_values["npubmeds"] > 0].corr(method="spearman"),
        file=sys.stderr,
    )
    print("length q_table", len(q_table), file=sys.stderr)
    print("length a_table", len(a_table), file=sys.stderr)
    pubmed_as = a_table.apply(lambda x: True if x["npubmeds"] > 0 else False, axis=1)
    print("As with pubmeds", len(pubmed_as[pubmed_as == True].index), file=sys.stderr)


def normalize_pmid(url, revisit_missing=True):
    """Convert various URLs to PMID using NCBI ID converter API

    Sometimes the PubMed URL has the title words instead of the PMID so we also have to 
    use the search API to retrieve the PMID.
    Also converts https to http (for compatibility)
    And other techniques were also implemented.

    :param url: url
    :type url: string
    :return: PMID or None if it could not be mapped
    :rtype: string or None

    :Example:
        >>> normalize_pmid("http://www.ncbi.nlm.nih.gov/pubmed?term=19404678")
        '19404678'
        >>> normalize_pmid("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2989813/")
        '21103316'
        >>> normalize_pmid("http://www.ncbi.nlm.nih.gov/pubmed?cmd=Link&dbFrom=PubMed&from_uid=15082451")
        '15082451'
        >>> normalize_pmid("http://www.ncbi.nlm.nih.gov/pmc/?cmd=Search&term=461182%5Bpmid%5D") # nao funciona
        ''
        >>> normalize_pmid("http://www.ncbi.nlm.nih.gov/pubmed?linkname=pubmed_pubmed_citedin&from_uid=2217192")
        2217192
        >>> http://dx.doi.org/10.1046/j.1469-8137.2002.00397.x
        ''
    """

    global pm_cache
    if url in pm_cache:
        return pm_cache[url]
    elif url in pm_cache["None"] and not revisit_missing:
        return None
    #elif revisit_missing is False:
    #    return None
    url_converter = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool={}&email={}&ids={}&format=json"
    pmsearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={}&api_key={}&format=json"
    if (
        url.strip("/").split("/")[-1] == "pmc"
        or url.strip("/").split("/")[-1] == "pubmed"
    ):  # links to PMC or PubMed homepage
        pm_cache["None"].add(url)
        return None
    print(url)
    if "pmc" in url and "pmid" not in url and "?term=" not in url and "cmd=search" not in url.lower():
        #print(url)
        # http://europepmc.org/backend/ptpmcrender.fcgi?accid=pmc1208485&blobtype=pdf
        # https://europepmc.org/article/pmc/pmc1208485
        #pmcid = url.split("/")[5][:10]
        #pmcid = "pmc" + url.split("pmc")[-1][:7]
        match = re.search(r"pmc([0-9]+)", url)
        #print(url, match, match.group(0))
        if match is None:
            pm_cache["None"].add(url)
            return None
        pmcid = match.group(0)
        # pmcid = pmcid.split("/")[0]
        try:
            result = requests.get(
                url_converter.format(params["toolname"], params["email"], pmcid)
            )
        except:
            pm_cache["None"].add(url)
            return None
        result = result.json()
        # print(result)
        if result["status"] == "error" or "pmid" not in result["records"][0]:
            # print("ERROR")
            # print(pmcid)
            # print(url)
            # print(result)
            pm_cache["None"].add(url)
            return None
        pmid = result["records"][0]["pmid"]
    elif ("pubmed" in url or "pmc" in url) and (
        "?term=" in url or "cmd=search" in url.lower()
    ):
        search_terms = urllib.parse.unquote(url.split("=")[-1])
        result = requests.get(pmsearch_url.format(search_terms, params["pubmed_api"]))
        result = result.json()
        if not result["esearchresult"]["idlist"]:
            # print("ERROR")
            # print(search_terms)
            # print(url)
            # print(result)
            pm_cache["None"].add(url)
            return None
        pmid = result["esearchresult"]["idlist"][0]
        # print("cmd/term search", url, pmid)
    elif "doi.org" in url:
        doi = "/".join(url.split("/")[3:])
        result = requests.get(
            url_converter.format(params["toolname"], params["email"], doi)
        )
        if "json" not in result.headers.get("Content-Type"):
            # print("Response content is not in JSON format.")
            # print("ERROR")
            # print(doi)
            # print(url)
            # print(result)
            pm_cache["None"].add(url)
            return None
        try:
            result = result.json()
        # print(result)
        except json.decoder.JSONDecodeError:
            print("error json decoder", result.text)
            pm_cache["None"].add(url)
            return None

        if result["status"] == "error" or "pmid" not in result["records"][0]:
            result = requests.get(
                pmsearch_url.format(doi + "[aid]", params["pubmed_api"])
            )
            result = result.json()
            if not result["esearchresult"]["idlist"]:
                # print("ERROR")
                # print(doi)
                # print(url)
                # print(result)
                pm_cache["None"].add(url)
                return None
            pmid = result["esearchresult"]["idlist"][0]
            # print("mapped doi with pm api", url, doi, pmid)
        else:
            pmid = result["records"][0]["pmid"]
            # print("mapped doi with id converter api", url, doi, pmid)
    elif "linkname=" in url.lower():
        pmid = url.split("=")[-1]
        # print("linkname=", url, pmid)
    elif "cmd=retrieve" in url.lower():
        match = re.search(r"([0-9]{8})", url)
        pmid = match.group()
    elif "/m/pubmed" in url.lower():
        pmid = url.split("/")[5]

    elif "artid=" in url.lower():
        pmcid = url.split("artid=")[-1].split("&")[0]
        response = requests.get(
            url_converter.format(params["toolname"], params["email"], "PMC" + pmcid)
        )
        result = response.text
        pmid = result.split("pmid=")[-1].split(" ")[0]


    elif "accid=" in url.lower():
        pmcid = url.split("accid=")[-1].split("&")[0]
        response = requests.get(
            url_converter.format(params["toolname"], params["email"], pmcid)
        )
        result = response.text
        pmid = result.split("pmid=")[-1].split(" ")[0]


    elif "pmid=" in url.lower():
        pmid = url.split("pmid=")[-1]

    elif "sciencedirect" in url.lower():
        r = requests.get(
            "https://api.elsevier.com/content/article/pii/"
            + url.split("pii/")[-1].split("?")[0]
            + "?apiKey={}".format(params["elsevier_api"])
        )
        response = r.text
        if "<pubmed-id>" in response:
            pmid = response.split("<pubmed-id>")[-1].split("</pubmed-id>")[0]
        else:
            pm_cache["None"].add(url)
            return None

    elif "researchgate" in url.lower():
        title = "+".join(url.lower().split("/")[-1].split("_")[1:])
        result = requests.get(pmsearch_url.format(title + "[title]", params["pubmed_api"]))
        #pmid = r.text.split("<Id>")[-1].split("</Id>")[0]
        try:
            result = result.json()
        except json.decoder.JSONDecodeError:
            print(result)
            print(url)
            return None
        if not result["esearchresult"]["idlist"]:
            #print("ERROR")
            # print(doi)
            #print(url)
            #print(result)
            pm_cache["None"].add(url)
            return None
        pmid = result["esearchresult"]["idlist"][0]


    elif (
        "imgur" in url.lower()
        or "youtube" in url.lower()
        or "book" in url.lower()
        or "projects" in url.lower()
        or "wiki" in url.lower()
        or ".jpg" in url.lower()
        or "flickr" in url.lower()
    ):
        pm_cache["None"].add(url)
        return None

    else:
        try:
            pmid = url.split("/")[4]
            # pmid = int(pmid)
            # pm_cache[url] = pmid

        except:
            pm_cache["None"].add(url)
            return None

    pmid = "".join([i for i in pmid if i.isdigit()])
    # pmid = "http://www.ncbi.nlm.nih.gov/pubmed/" + pmid
    print(pmid)
    pm_cache[url] = pmid
    return pmid
