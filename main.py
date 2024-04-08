import sys

import IRsys as ir

# initializes the necessary structures to perform multiple searches later
positional_index = ir.build_positional_index()[0]
file_map = ir.build_positional_index()[1]
N = len(file_map)
tfidf_matrix = ir.build_tfidf_matrix(positional_index, N)
doc_lengths = ir.calc_doc_length(tfidf_matrix, N)
normalized_terms = ir.normalize_terms(tfidf_matrix, doc_lengths)

# Print the positional index and the tf.idf matrix
ir.show_data(positional_index, tfidf_matrix, file_map, normalized_terms)


# takes queries from users and returns matches results ranked, if any
choice = 1
while choice == 1:
    query = input("Enter your search query:\n")
    ranked_results = ir.engine(query, positional_index, file_map, tfidf_matrix, doc_lengths, N)
    if ranked_results == {}:
        print("There are no matching documents for your search query")
    else:
        print("  Search results ranked\n")
        i = 1
        for result in ranked_results:
            print(str(i) + "." + " Document name: " + str(result) + "\n   Cosine similarity: "
                  + str(round(ranked_results[result], 3)) + "\n")
            i += 1
    try:
        choice = int(input("enter 1 for another search or 0 to exit\n"))
    except ValueError:
        print("invalid entry")
        sys.exit()



