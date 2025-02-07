needle_dict = {
    'SF':{
        'needle': 'The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day. ',
        'retrieval_question': 'What is the best thing to do in San Francisco?',
        'substr_validation_words': 'eat,sandwich,sit,dolores,park,sunny,day'
    },
    'SF-indicator':{
        'needle': 'The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day. ',
        'retrieval_question': 'What is the best thing to do in San Francisco? [IND]',
        'substr_validation_words': 'eat,sandwich,sit,dolores,park,sunny,day'
    },
    'pizza':{
        'needle': 'The secret ingredients needed to build the perfect pizza are figs, prosciutto, smoked applewood bacon, lemon, goat cheese, truffle honey, pear slices, espresso-soaked dates, gorgonzola dolce, and candied walnuts. ',
        'retrieval_question': 'What are the secret ingredients needed to build the perfect pizza?',
        'substr_validation_words': 'figs,prosciutto,smoked,applewood,bacon,lemon,goat,cheese,truffle,honey,pear,slices,espresso,soaked,dates,gorgonzola,dolce,candied,walnuts'
    },
    'pizza-indicator':{
        'needle': 'The secret ingredients needed to build the perfect pizza are figs, prosciutto, smoked applewood bacon, lemon, goat cheese, truffle honey, pear slices, espresso-soaked dates, gorgonzola dolce, and candied walnuts. ',
        'retrieval_question': 'What are the secret ingredients needed to build the perfect pizza? [IND]',
        'substr_validation_words': 'figs,prosciutto,smoked,applewood,bacon,lemon,goat,cheese,truffle,honey,pear,slices,espresso,soaked,dates,gorgonzola,dolce,candied,walnuts'
    },
    'pizza-short':{
        'needle': 'The secret ingredients needed to build the perfect pizza are figs, prosciutto, smoked applewood bacon and lemon. ',
        'retrieval_question': 'What are the secret ingredients needed to build the perfect pizza?',
        'substr_validation_words': 'figs,prosciutto,smoked,applewood,bacon,lemon'
    },
}

template_dict = {
    "initial": "You are a helpful AI assistant that answers a question using only the provided document: \n{context}\n\nQuestion: {retrieval_question}",
    "1": "You are a helpful AI assistant that answers a question using only the provided context: \n{context}\n\nQuestion: {retrieval_question}",
    "2": "Document: \n{context}\n\nAnswer the question accoriding to the provided document: {retrieval_question}",
    "3": "Context: \n{context}\n\nAnswer the question accoriding to the provided context: {retrieval_question}",
}
