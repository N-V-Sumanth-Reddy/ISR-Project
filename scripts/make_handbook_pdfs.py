"""Build the two PDF documents of the fictional demo handbook (needs `pip install reportlab`).

The PDFs are already in data/handbook/docs; this script only exists so they can be regenerated.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "data" / "handbook" / "docs"
NOTE = "Fictional sample document written for the RAG demo."

TRAVEL = ("Travel and Expense Reimbursement Policy", "Brightwater Institute - Office of the Controller Policy FIN-220.", [
    [("1. Authorization before travel", [
        "All overnight travel on Institute business requires an approved Travel Authorization. The request is "
        "submitted at least 10 business days before departure. International travel is submitted at least 30 "
        "calendar days before departure, because it also goes through an export-control review.",
        "Travel is approved by the traveler's department head. Travel by a department head is approved by the dean.",
      ]),
      ("2. Air travel", [
        "Airfare is booked in economy class through the Institute travel portal, TripDesk. Premium economy is "
        "allowed only for flights with a scheduled flight time of more than 8 hours. Business class and first "
        "class are not reimbursed.",
        "Airline club memberships, seat upgrades and travel insurance bought by the traveler are personal expenses.",
      ])],
    [("3. Lodging", [
        "Lodging for domestic travel is reimbursed at actual cost up to 180 dollars per night before taxes. In "
        "designated high-cost cities the limit is 240 dollars per night. The designated high-cost cities are New "
        "York, San Francisco, Boston and Washington, D.C.",
        "A traveler who stays at the official conference hotel may be reimbursed at the conference rate when the "
        "department head approves it in advance.",
      ]),
      ("4. Meals", [
        "Meals during domestic travel are paid as a per diem of 64 dollars per day. On the first and the last "
        "day of a trip the per diem is paid at 75 percent. Receipts are not required for per diem meals. Alcohol "
        "is never reimbursed.",
      ]),
      ("5. Ground transportation", [
        "Use of a personal vehicle is reimbursed at 0.62 dollars per mile. Rental cars are limited to the "
        "mid-size class or smaller. Traffic and parking fines are the traveler's responsibility.",
      ])],
    [("6. Expense reports", [
        "An expense report is submitted within 30 days of the end of the trip. Reimbursements claimed more than "
        "60 days after the trip are reported as taxable income to the traveler. Expenses submitted more than 90 "
        "days after the trip are not reimbursed.",
        "A receipt is required for every expense of 40 dollars or more, other than per diem meals.",
      ]),
      ("7. Other rules", [
        "Conference registration fees may be paid with a purchasing card. Personal entertainment, such as "
        "in-room movies, is not reimbursed. Travelers who combine personal and business travel pay any extra "
        "cost that results from the personal portion.",
      ])],
])

DATA = ("Research Data Classification and Handling Standard", "Brightwater Institute - Information Security Office Standard IS-201.", [
    [("1. Classification levels", [
        "Institute data is assigned to one of four levels: Public (Level 1), Internal (Level 2), Confidential "
        "(Level 3) and Restricted (Level 4). Data that has not been classified is treated as Internal.",
        "The principal investigator is the data steward for a research project and decides the classification of "
        "the project's data.",
      ]),
      ("2. Restricted data (Level 4)", [
        "Restricted data includes identifiable health information, Social Security numbers, export-controlled "
        "technical data and payment card data. Restricted data is stored only in the Vault secure enclave, where "
        "it is encrypted at rest with AES-256. It may not be stored on personal devices and may not be sent by "
        "email. Access lists for Restricted data are reviewed every 90 days.",
      ])],
    [("3. Confidential data (Level 3)", [
        "Confidential data includes student education records, personnel files, and unpublished research data "
        "covered by a partner's non-disclosure agreement. Confidential data is stored on BrightDrive, the "
        "Institute-managed storage service. It may not be placed in personal cloud storage accounts. Sharing "
        "Confidential data outside the Institute requires a signed data use agreement.",
      ]),
      ("4. Use of AI tools", [
        "Confidential and Restricted data may not be entered into externally hosted AI tools, including public "
        "chatbots. Such data may be processed only with AI tools that the Information Security Office has "
        "approved and that are hosted locally on Institute systems.",
      ])],
    [("5. Incidents", [
        "A suspected exposure of Confidential or Restricted data is reported to the Information Security Office "
        "within 24 hours, by calling the security hotline at extension 5-7300.",
      ]),
      ("6. Retention and disposal", [
        "Research data is kept for at least seven years after publication or after the end of the project, "
        "whichever is later. Media that held Restricted data is destroyed by certified shredding arranged by "
        "Facilities, and the certificate of destruction is kept for three years.",
      ])],
])


def build(spec, filename: str) -> None:
    title, owner, pages = spec
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUT / filename), pagesize=letter, title=title, author="Brightwater Institute",
                            leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch)
    story = [Paragraph(title, styles["Title"]), Paragraph(owner + " " + NOTE, styles["Italic"]), Spacer(1, 14)]
    for i, page in enumerate(pages):
        if i:
            story.append(PageBreak())
        for heading, paras in page:
            story.append(Paragraph(heading, styles["Heading2"]))
            for para in paras:
                story.append(Paragraph(para, styles["BodyText"]))
                story.append(Spacer(1, 8))
    doc.build(story)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build(TRAVEL, "travel_and_expense_policy.pdf")
    build(DATA, "research_data_classification_standard.pdf")
    print("wrote 2 PDFs to", OUT)
