from __future__ import print_function
import re

import mechanize
import requests


def main():
    url = "http://nesssi.cacr.caltech.edu/cgi-bin/getcssconedb_id.cgi"
    payload = {'id': 1001006023682,
               'ID': 1001006023682,
               'output_type': "html",
               'output_format': "short"}
    #
    # session = requests.session()
    # r = requests.post(url, params=payload)
    # # with open("requests_results.html", "w") as f:
    # #     f.write(r.content)
    #
    # print(r.content)
    # print(r.cookies)

    br = mechanize.Browser()
    br.set_handle_robots(False)
    br.open(url)

    frm = br.forms()[0]

    print(frm.attrs)
    br.select_form(nr=0)
    br.form["ID"] = 1001006023682
    req = br.submit()
    print(req)


if __name__ == "__main__":
    main()
