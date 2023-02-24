import spacy
nlp = spacy.load('en_core_web_md')

with open('movies.txt', 'r') as f:
    movies = f.readlines()
    sample = ''''Planet Hulk: Will he save their world or destroy it? When the Hulk becomes too dangerous for the
                Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a
                planet where the Hulk can live in peace. Unfortunately, Hulk land on the
                planet Sakaar where he is sold into slavery and trained as a gladiator.'''


def recommended_movie(movies, sample):
    max_sim = 0
    recommended = ' '
    model = nlp(sample)
    for movie in movies:
        similarity = nlp(movie).similarity(model)
        if similarity > max_sim:
            max_sim = similarity
            recommended = movie.split(':')[0]
    return recommended.strip()

recommended = recommended_movie(movies, sample)
print(recommended)