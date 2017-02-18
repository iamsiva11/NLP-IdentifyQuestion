# Identify Question Type - NLP

Given a question, the aim is to identify the category it belongs to. The four 
categories: Who, What, When, Affirmation(yes/no).
Label any sentence that does not fall in any of the above four as "Unknown" type.

Example:
*  What is your name? Type: What
*  When is the show happening? Type: When
*  Is there a cab available for airport? Type: Affirmation
*  There are ambiguous cases to handle as well like: What time does the train leave(this looks like a what question but is actually a When type)

