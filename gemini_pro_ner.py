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


# The same comparison function but disregards any position

def compare_columns_ignore_position(text1, text2):
    # Function to extract rows from a string
    def extract_rows(text):
        return [line for line in text.strip().split('\n') if line.strip()]

    # Get the rows for both texts
    rows1 = extract_rows(text1)
    rows2 = extract_rows(text2)

    # Extract the last words from each row
    last_words1 = {row.split()[-1] for row in rows1}
    last_words2 = {row.split()[-1] for row in rows2}

    # Find common last words
    common_words = last_words1 & last_words2

    # Calculate matches and total unique last words
    matches = len(common_words)
    total_unique_words = len(last_words1 | last_words2)

    # Calculate match percentage
    match_percentage = (matches / total_unique_words) * 100 if total_unique_words > 0 else 0

    # Find missing last words
    missing_in_text1 = last_words2 - last_words1
    missing_in_text2 = last_words1 - last_words2

    return {
        "match_percentage": match_percentage,
        "common_words": list(common_words),
        "missing_in_text1": list(missing_in_text1),
        "missing_in_text2": list(missing_in_text2),
    }



# a function to extract the first column  of the labeled text to make it easier to prompt the model without repeating the text with and without labels

def extract_first_column(input_text):
    """
    Extracts the first column from the given text input.

    Args:
        input_text (str): Multiline string where each line contains columns separated by spaces.

    Returns:
        str: Multiline string containing only the first column of the input.
    """
    lines = input_text.strip().split("\n")  # Split input into lines
    first_column = [line.split()[0] for line in lines]  # Extract the first column
    return "\n".join(first_column)


# A correctly labeled text taken directly from the hand made training data of the ML4OPT NER task used for accuracy measurement

text1 = """An	_	_	O
electronics	_	_	O
store	_	_	O
wants	_	_	O
to	_	_	O
optimize	_	_	O
how	_	_	O
many	_	_	O
phones	_	_	B-VAR
and	_	_	O
laptops	_	_	B-VAR
are	_	_	O
enough	_	_	O
to	_	_	O
keep	_	_	O
in	_	_	O
inventory	_	_	O
.	_	_	O
A	_	_	O
phone	_	_	B-VAR
will	_	_	O
earn	_	_	O
the	_	_	O
store	_	_	O
$	_	_	O
120	_	_	B-PARAM
in	_	_	O
profits	_	_	B-OBJ_NAME
,	_	_	O
and	_	_	O
a	_	_	O
laptop	_	_	B-VAR
will	_	_	O
earn	_	_	O
$	_	_	O
40	_	_	B-PARAM
.	_	_	O
A	_	_	O
phone	_	_	B-VAR
requires	_	_	O
1	_	_	B-PARAM
sq	_	_	O
ft	_	_	O
of	_	_	O
floor	_	_	O
space	_	_	O
,	_	_	O
whereas	_	_	O
a	_	_	O
laptop	_	_	B-VAR
requires	_	_	O
4	_	_	B-PARAM
sq	_	_	O
ft	_	_	O
.	_	_	O
In	_	_	O
total	_	_	O
,	_	_	O
400	_	_	B-LIMIT
sq	_	_	O
ft	_	_	O
of	_	_	O
floor	_	_	O
space	_	_	O
is	_	_	O
available	_	_	B-CONST_DIR
.	_	_	O
The	_	_	O
store	_	_	O
stocks	_	_	O
only	_	_	O
phones	_	_	B-VAR
and	_	_	O
laptops	_	_	B-VAR
.	_	_	O
Corporate	_	_	O
has	_	_	O
required	_	_	O
that	_	_	O
at	_	_	B-CONST_DIR
least	_	_	I-CONST_DIR
80	_	_	B-LIMIT
%	_	_	I-LIMIT
of	_	_	O
all	_	_	O
appliances	_	_	O
in	_	_	O
stock	_	_	O
be	_	_	O
laptops	_	_	B-VAR
.	_	_	O
Finally	_	_	O
,	_	_	O
a	_	_	O
phone	_	_	B-VAR
costs	_	_	O
$	_	_	O
400	_	_	B-PARAM
for	_	_	O
the	_	_	O
store	_	_	O
,	_	_	O
and	_	_	O
a	_	_	O
laptop	_	_	B-VAR
,	_	_	O
$	_	_	O
100	_	_	B-PARAM
.	_	_	O
The	_	_	O
store	_	_	O
wants	_	_	O
to	_	_	O
spend	_	_	O
at	_	_	B-CONST_DIR
most	_	_	I-CONST_DIR
$	_	_	O
6000	_	_	B-LIMIT
.	_	_	O
Formulate	_	_	O
an	_	_	O
LP	_	_	O
that	_	_	O
can	_	_	O
be	_	_	O
used	_	_	O
to	_	_	O
maximize	_	_	B-OBJ_DIR
the	_	_	O
store	_	_	O
's	_	_	O
profit	_	_	B-OBJ_NAME
.	_	_	O
 """


# ========================================= Prompt construction ======================================

# the reply example in the prompt

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

# the example unlabeled text for the one shot example (extracted from the labeled reply text above)
example_text = extract_first_column(reply_text)


# the text to be labeled by the LLM (extracted from the labeled text1 above)
prompt_text = extract_first_column(text1)


# the prompt where all the strings above are used
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
        temperature=1.5,
        top_p= 0.95,
        top_k= 1,
        max_output_tokens=2048
    ),
)
print(response.text)


# result = compare_columns_with_missing_rows(text1=text1, text2=str(response.text)) # uncomment to check the results without ignoring the position of the rows
result = compare_columns_ignore_position(text1=text1, text2=str(response.text))
print(f"Match Percentage: {result['match_percentage']:.2f}%")
print(f"Missing in Text1: {result['missing_in_text1']}")
print(f"Missing in Text2: {result['missing_in_text2']}")
# print(f"Non-Matching Rows: {result['non_matching_rows']}") # uncomment in case of using the comparison function which doesn't ignore the positions of the rows



# ========================================================   this area is for testing purposes ===========================================

# a randomized prompt to check if order of the words affect model accuracy


# randomized_prompt_text = """
# Corporate
# has
# required
# that
# at
# least
# 80
# %
# of
# all
# appliances
# in
# stock
# be
# laptops
# .
# The
# store
# stocks
# only
# phones
# and
# laptops
# .
# A
# phone
# requires
# 1
# sq
# ft
# of
# floor
# space
# ,
# whereas
# a
# laptop
# requires
# 4
# sq
# ft
# .
# In
# total
# ,
# 400
# sq
# ft
# of
# floor
# space
# is
# available
# .
# A
# phone
# will
# earn
# the
# store
# $
# 120
# in
# profits
# ,
# and
# a
# laptop
# will
# earn
# $
# 40
# .
# Finally
# ,
# a
# phone
# costs
# $
# 400
# for
# the
# store
# ,
# and
# a
# laptop
# ,
# $
# 100
# .
# The
# store
# wants
# to
# spend
# at
# most
# $
# 6000
# .
# An
# electronics
# store
# wants
# to
# optimize
# how
# many
# phones
# and
# laptops
# are
# enough
# to
# keep
# in
# inventory
# .
# Formulate
# an
# LP
# that
# can
# be
# used
# to
# maximize
# the
# store
# 's
# profit
# .
#  """