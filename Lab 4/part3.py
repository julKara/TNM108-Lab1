text = "Neo-Nazism consists of post-World War II militant social or political movements seeking to revive and implement the ideology of Nazism. Neo-Nazis seek to employ their ideology to promote hatred and attack minorities, or in some cases to create a fascist political state. It is a global phenomenon, with organizedrepresentation in many countries and international networks. It borrows elementsfrom Nazi doctrine, including ultranationalism, racism, xenophobia, ableism, homophobia, anti-Romanyism, antisemitism, anti-communism and initiating the FourthReich. Holocaust denial is a common feature, as is the incorporation of Nazi symbols and admiration of Adolf Hitler. In some European and Latin American countries, laws prohibit the expression of pro-Nazi, racist, anti-Semitic, or homophobic views. Many Nazi-related symbols are banned in European countries (especiallyGermany) in an effort to curtail neo-Nazism. The term neo-Nazism describes any post-World War II militant, social or political movements seeking to revive the ideology of Nazism in whole or in part. The term neo-Nazism can also refer to theideology of these movements, which may borrow elements from Nazi doctrine, including ultranationalism, anti-communism, racism, ableism, xenophobia, homophobia,anti-Romanyism, antisemitism, up to initiating the Fourth Reich. Holocaust denial is a common feature, as is the incorporation of Nazi symbols and admiration ofAdolf Hitler. Neo-Nazism is considered a particular form of far-right politics and right-wing extremism."

from summa.summarizer import summarize

# Define length of the summary as a proportion of the text
print(summarize(text, ratio=0.2))   # Summerize

# Summerization with specified length
summarize(text, words=50) # Same as earlier

# Keyword extraction
from summa import keywords

print("Keywords:\n",keywords.keywords(text))
# to print the top 3 keywords
print("Top 3 Keywords:\n",keywords.keywords(text,words=3))

####### Our little test #######
text2 = "Sol has brown hair and eyes, which are sometimes depicted as reddish-brown or golden with slit pupils—the latter of which is to distinguish his true nature as a Gear. Wearing a red headband bearing the words 'Rock You', Sol keeps his spiky, waist-length hair tied at all times. The headband conceals the red brand on his forehead that would expose him as a Gear. He has muscular physique, whether he is infused with Flame of Corruption or not. As Frederick, he is depicted with either brown or blue eyes; he sports a much shorter hair, a muscular physique and a white labcoat. He wears a tight, black undershirt with the top half covered by a red sleeveless jacket, along with white jeans. His belt buckle has the word 'FREE' scratched on it and a drape of red cloth hanging down beneath. Sol also wears a pair of black fingerless gloves, red shoes, and a multitude of belts, including a lone one on his left bicep. In later games, he is seen sporting long-sleeved, red leather jackets with handcuffs strapped in the sides of the sleeves and a single black accent at the back of his jackets as well."

# Define length of the summary as a proportion of the text
print(summarize(text2, ratio=0.2))   # Summerize

# Summerization with specified length
print(summarize(text2, words=50)) # Same as earlier

print("Keywords:\n",keywords.keywords(text2))
# to print the top 3 keywords
print("Top 3 Keywords:\n",keywords.keywords(text2,words=3))