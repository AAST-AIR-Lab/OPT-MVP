import google.generativeai as genai
GOOGLE_API_KEY = "<Your Google API Key>"


# basic functions for output accuracy measurement

def compare_columns_with_missing_rows(text1, text2):
    # Function to extract rows from a string
    def extract_rows(text):
        return [line for line in text.strip().split('\n') if line.strip()]

    # Get the rows for both texts
    rows1 = extract_rows(text1)
    rows2 = extract_rows(text2)

    # Find missing rows in each text
    missing_in_text1 = rows2[len(rows1):] if len(rows1) < len(rows2) else []
    missing_in_text2 = rows1[len(rows2):] if len(rows2) < len(rows1) else []

    # Determine the minimum number of rows to compare
    min_rows = min(len(rows1), len(rows2))

    # Extract the words in the last column for comparison
    words1 = [rows1[i].split()[-1] for i in range(min_rows)]
    words2 = [rows2[i].split()[-1] for i in range(min_rows)]

    # Calculate matches and find non-matching rows
    matches = 0
    non_matching_rows = []

    for i, (word1, word2) in enumerate(zip(words1, words2)):
        if word1 == word2:
            matches += 1
        else:
            non_matching_rows.append((rows1[i], rows2[i]))

    # Calculate percentage of matches for the comparable rows
    match_percentage = (matches / min_rows) * 100 if min_rows > 0 else 0

    return {
        "match_percentage": match_percentage,
        "non_matching_rows": non_matching_rows,
        "missing_in_text1": missing_in_text1,
        "missing_in_text2": missing_in_text2,
    }


# A correctly labeled text taken directly from the hand made training data of the ML4OPT NER task used for accuracy measurement

text1 = """Nova	_	_	O
Network	_	_	O
wants	_	_	O
to	_	_	O
design	_	_	O
a	_	_	O
plan	_	_	O
to	_	_	O
bid	_	_	O
for	_	_	O
the	_	_	O
job	_	_	O
of	_	_	O
providing	_	_	O
a	_	_	O
computer	_	_	O
network	_	_	O
for	_	_	O
city	_	_	O
offices	_	_	O
.	_	_	O
It	_	_	O
can	_	_	O
build	_	_	O
three	_	_	O
types	_	_	O
of	_	_	O
layouts	_	_	O
using	_	_	O
workstations	_	_	O
,	_	_	O
servers	_	_	O
,	_	_	O
and	_	_	O
switches	_	_	O
.	_	_	O
It	_	_	O
has	_	_	B-CONST_DIR
2000	_	_	B-LIMIT
workstations	_	_	O
,	_	_	O
500	_	_	B-LIMIT
servers	_	_	O
,	_	_	O
and	_	_	O
300	_	_	B-LIMIT
switches	_	_	O
.	_	_	O
A	_	_	O
ring	_	_	B-VAR
layout	_	_	I-VAR
uses	_	_	O
50	_	_	B-PARAM
workstations	_	_	O
,	_	_	O
20	_	_	B-PARAM
servers	_	_	O
,	_	_	O
and	_	_	O
10	_	_	B-PARAM
switches	_	_	O
;	_	_	O
a	_	_	O
tree	_	_	B-VAR
layout	_	_	I-VAR
uses	_	_	O
30	_	_	B-PARAM
workstations	_	_	O
,	_	_	O
15	_	_	B-PARAM
servers	_	_	O
,	_	_	O
and	_	_	O
7	_	_	B-PARAM
switches	_	_	O
;	_	_	O
and	_	_	O
a	_	_	O
mesh	_	_	B-VAR
layout	_	_	I-VAR
uses	_	_	O
100	_	_	B-PARAM
workstations	_	_	O
,	_	_	O
50	_	_	B-PARAM
servers	_	_	O
,	_	_	O
and	_	_	O
30	_	_	B-PARAM
switches	_	_	O
.	_	_	O
The	_	_	O
net	_	_	O
profit	_	_	B-OBJ_NAME
is	_	_	O
$	_	_	O
2000	_	_	B-PARAM
for	_	_	O
each	_	_	O
ring	_	_	B-VAR
layout	_	_	I-VAR
,	_	_	O
$	_	_	O
4000	_	_	B-PARAM
for	_	_	O
each	_	_	O
tree	_	_	B-VAR
layout	_	_	I-VAR
,	_	_	O
and	_	_	O
$	_	_	O
8000	_	_	B-PARAM
for	_	_	O
each	_	_	O
mesh	_	_	B-VAR
layout	_	_	I-VAR
.	_	_	O
How	_	_	O
many	_	_	O
layouts	_	_	O
of	_	_	O
each	_	_	O
type	_	_	O
should	_	_	O
be	_	_	O
used	_	_	O
to	_	_	O
yield	_	_	O
maximum	_	_	B-OBJ_DIR
profit	_	_	B-OBJ_NAME
?	_	_	O
 """


# ========================================= Prompt construction ======================================

example_text = """
Cautious
Asset
Investment
has
a
total
of
$
150,000
to
manage
and
decides
to
invest
it
in
money
market
fund
,
which
yields
a
2
%
return
as
well
as
in
foreign
bonds
,
which
gives
and
average
rate
of
return
of
10.2
%
.
Internal
policies
require
PAI
to
diversify
the
asset
allocation
so
that
the
minimum
investment
in
money
market
fund
is
40
%
of
the
total
investment
.
Due
to
the
risk
of
default
of
foreign
countries
,
no
more
than
40
%
of
the
total
investment
should
be
allocated
to
foreign
bonds
.
How
much
should
the
Cautious
Asset
Investment
allocate
in
each
asset
so
as
to
maximize
its
average
return
?
"""

reply_text = """
Cautious	_	_	O
Asset	_	_	O
Investment	_	_	O
has	_	_	O
a	_	_	O
total	_	_	B-CONST_DIR
of	_	_	O
$	_	_	O
150,000	_	_	B-LIMIT
to	_	_	O
manage	_	_	O
and	_	_	O
decides	_	_	O
to	_	_	O
invest	_	_	O
it	_	_	O
in	_	_	O
money	_	_	B-VAR
market	_	_	I-VAR
fund	_	_	I-VAR
,	_	_	O
which	_	_	O
yields	_	_	O
a	_	_	O
2	_	_	B-PARAM
%	_	_	I-PARAM
return	_	_	B-OBJ_NAME
as	_	_	O
well	_	_	O
as	_	_	O
in	_	_	O
foreign	_	_	B-VAR
bonds	_	_	I-VAR
,	_	_	O
which	_	_	O
gives	_	_	O
and	_	_	O
average	_	_	O
rate	_	_	O
of	_	_	O
return	_	_	B-OBJ_NAME
of	_	_	O
10.2	_	_	B-PARAM
%	_	_	O
.	_	_	O
Internal	_	_	O
policies	_	_	O
require	_	_	O
PAI	_	_	O
to	_	_	O
diversify	_	_	O
the	_	_	O
asset	_	_	O
allocation	_	_	O
so	_	_	O
that	_	_	O
the	_	_	O
minimum	_	_	B-CONST_DIR
investment	_	_	O
in	_	_	O
money	_	_	B-VAR
market	_	_	I-VAR
fund	_	_	I-VAR
is	_	_	O
40	_	_	B-LIMIT
%	_	_	I-LIMIT
of	_	_	O
the	_	_	O
total	_	_	O
investment	_	_	O
.	_	_	O
Due	_	_	O
to	_	_	O
the	_	_	O
risk	_	_	O
of	_	_	O
default	_	_	O
of	_	_	O
foreign	_	_	O
countries	_	_	O
,	_	_	O
no	_	_	B-CONST_DIR
more	_	_	I-CONST_DIR
than	_	_	I-CONST_DIR
40	_	_	B-LIMIT
%	_	_	I-LIMIT
of	_	_	O
the	_	_	O
total	_	_	O
investment	_	_	O
should	_	_	O
be	_	_	O
allocated	_	_	O
to	_	_	O
foreign	_	_	B-VAR
bonds	_	_	I-VAR
.	_	_	O
How	_	_	O
much	_	_	O
should	_	_	O
the	_	_	O
Cautious	_	_	O
Asset	_	_	O
Investment	_	_	O
allocate	_	_	O
in	_	_	O
each	_	_	O
asset	_	_	O
so	_	_	O
as	_	_	O
to	_	_	O
maximize	_	_	B-OBJ_DIR
its	_	_	O
average	_	_	B-OBJ_NAME
return	_	_	I-OBJ_NAME
?	_	_	O
"""

prompt_text = """
John
has
a
300
acre
berry
farm
on
which
to
plant
blueberries
and
raspberries
.
John
has
$
10000
to
spend
on
watering
and
575
days
worth
of
labor
available
.
For
each
acre
of
blueberries
,
6
days
worth
of
labor
and
$
22
in
watering
costs
is
required
.
For
each
acre
of
raspberries
,
3
days
worth
of
labor
and
$
25
in
watering
costs
is
required
.
The
profit
per
acre
of
blueberries
is
$
56
and
the
profit
per
acre
of
raspberries
is
$
75
.
Formulate
an
LP
problem
in
order
to
maximize
profit
.
 """

prompt_text = """
Nova
Network
wants
to
design
a
plan
to
bid
for
the
job
of
providing
a
computer
network
for
city
offices
.
It
can
build
three
types
of
layouts
using
workstations
,
servers
,
and
switches
.
It
has
2000
workstations
,
500
servers
,
and
300
switches
.
A
ring
layout
uses
50
workstations
,
20
servers
,
and
10
switches
;
a
tree
layout
uses
30
workstations
,
15
servers
,
and
7
switches
;
and
a
mesh
layout
uses
100
workstations
,
50
servers
,
and
30
switches
.
The
net
profit
is
$
2000
for
each
ring
layout
,
$
4000
for
each
tree
layout
,
and
$
8000
for
each
mesh
layout
.
How
many
layouts
of
each
type
should
be
used
to
yield
maximum
profit
?
 """

context_text = f"""
## USER: the following text is an example of a text description of a problem that requires optimization your role is to classify each word should be classified into one of the following: (general word refered to using the label 'O' , optimization_problem_constraint_description refered to using the label 'B/I-CONST_DIR', numerical_limit refered to using the label 'B/I-LIMIT',variable refered to using the label 'B/I-VAR', measurable_parameter refered to using the label 'B/I PARAM', objective refered to using the label 'B/I-OBJ_NAME' , action_towards_objective refered to using the label 'B-OBJ_DIR') answer in the format 'word _ _ classification label' for each word in the provided text
## provided text:
{example_text}
## Assistant:
{reply_text}
## USER: the following text is an example of a text description of a problem that requires optimization your role is to classify each word should be classified into one of the following: (general word refered to using the label 'O' , optimization_problem_constraint_description refered to using the label 'B/I-CONST_DIR', numerical_limit refered to using the label 'B/I-LIMIT',variable refered to using the label 'B/I-VAR', measurable_parameter refered to using the label 'B/I PARAM', objective refered to using the label 'B/I-OBJ_NAME' , action_towards_objective refered to using the label 'B-OBJ_DIR') answer in the format 'word _ _ classification label' for each word in the provided text
{prompt_text}
## Assistant:
"""

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(
    "models/gemini-1.5-pro",
    system_instruction="You are a helpful, smart and respectful assistant who always answers based on the provided information while thinking logically step by step",
)
response = model.generate_content(
    context_text,
    generation_config=genai.types.GenerationConfig(
        # Only one candidate for now.
        candidate_count=1,
        temperature=1.0,
        top_p= 0.95,
        top_k= 1,
        max_output_tokens=2048
    ),
)
print(response.text)


result = compare_columns_with_missing_rows(text1, str(response.text))

print(f"Match Percentage: {result['match_percentage']:.2f}%")
print(f"Non-Matching Rows: {result['non_matching_rows']}")
print(f"Missing in Text1: {result['missing_in_text1']}")
print(f"Missing in Text2: {result['missing_in_text2']}")