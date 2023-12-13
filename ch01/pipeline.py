from transformers import pipeline

def run_pipelines(types):
    if 'sentiment' in types:
        sentiment = pipeline('sentiment-analysis')
        res = sentiment('During the 11th Annual Interactive Achievement Awards, the Academy of Interactive Arts and Sciences named The Witcher as one of the nominees for 2007\'s Role-Playing Game of the Year; however, it was eventually awarded to Mass Effect.')
        print('>>>', res)

    if 'zero-shot' in types:
        labels = ['book', 'fantasy', 'videogame', 'Andrzej Sapkowski', 'CD Projekt Red']
        classifier = pipeline('zero-shot-classification')
        res = classifier(
            'Witchers are humans taken as children, trained, and physically mutated for the purpose of killing monsters',
            candidate_labels=labels
        )
        pairs = zip(res['labels'], res['scores'])
        print('>>>', [ (label, score) for (label, score) in pairs ])

    if 'generation' in types:
        generator = pipeline('text-generation', model='distilgpt2')
        res = generator('Witchers are', num_return_sequences=1, max_length=140)
        print('>>>', res)

    if 'ner' in types:
        ner = pipeline('ner', aggregation_strategy='simple')
        res = ner('Witchers are humans like Geralt taken as children, trained, and physically mutated for the purpose of killing monsters, they are trained in Kaer Morhen')
        print('>>>', res)

    if 'qa' in types:
        question_answerer = pipeline("question-answering")
        res = question_answerer(
            question='Are witchers adults?',
            context='Witchers are humans taken as children, trained, and physically mutated for the purpose of killing monsters',
        )
        print('>>>', res)

run_pipelines('qa')
