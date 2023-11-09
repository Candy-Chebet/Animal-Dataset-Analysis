import numpy
from data import load_dataset # data is included with midterm
from data import print_tree
from data import decision_path_for_animal
from data import print_path_for_animal
from sklearn import tree

"""helper functions"""

def distance(x,y):
    """return the sum-squared-difference between x and y"""
    diff = x-y
    return diff.T@diff

def count(x):
    """return the number of 1's appearing in x"""
    return numpy.sum(x == 1)
    #return np.sum(a) # this also works if x is 0's and 1's

def entropy(x,y):
    """entropy reduction (information gain) for a feature covering x out
    of y animals"""
    Hy = numpy.log2(y)
    Hyx = (x/y)*numpy.log2(x) + ((y-x)/y)*numpy.log2(y-x)
    return Hy - Hyx

# load the dataset
animals_data = load_dataset()
labels,features,data = animals_data
data = numpy.array(data)
N = len(labels)   # number of animals
M = len(features) # number of features

# first, find the two most similar animals
# consider the first pair, 0 & 1 and compute their distance
min_i,min_j = 0,1
min_dist = distance(data[min_i],data[min_j])

# iterate over every pair (i,j) of animals
for i in range(N):
    for j in range(i+1,N):
        dist = distance(data[i],data[j])
        if dist < min_dist:
            # if we find a more similar pair, save it
            min_dist = dist
            min_i,min_j = i,j

# print them
print("=== most similar animals ===")
print("  %s (id:%d)" % (labels[min_i],min_i))
print("  %s (id:%d)" % (labels[min_j],min_j))
print("  differs in %d (out of %d) features" % (min_dist,M))
print()

print("=== feature counts ===")
for i in range(3):
    feature_name = features[i]
    feature_column = data[:,i]
    f_count = count(feature_column)
    print("feature %s has %d/%d yes answers" % (feature_name,f_count,N))
print("...")
print()

print("=== training decision tree  ===")
decision_tree = tree.DecisionTreeClassifier(criterion='entropy',
                                            random_state=4412)
decition_tree = decision_tree.fit(data,labels)
pred_labels = decition_tree.predict(data)
acc = 100*sum(labels == pred_labels)/len(labels)
print("accuracy: %.4f%%" % acc)
print()

print("=== printing decision tree ===")
print("    (uncomment in code to view)")
#print_tree(decision_tree,animals_data)
print()

animal = "wolf"
print("=== decision path for %s ===" % animal)
print("    (chosen branch in brackets)")
path = decision_path_for_animal(decision_tree,animal,animals_data)
print_path_for_animal(decision_tree,path,animal,animals_data)
print("path has length %d" % (len(path)-1))
def load_matrix(filename):
    """load a data matrix from file as a dictionary"""
    with open(filename, 'r') as f:
        lines = f.readlines()

    animals_data = {}  # Dictionary to store animal data
    lines = [line.strip().split(' ') for line in lines]
    for line in lines:
        animal_label = line[0]
        features = list(map(int, line[1:]))
        animals_data[animal_label] = features

    return animals_data
def shortest_and_longest_paths(decision_tree, animals_data):
    def find_paths(decision_tree, animals_data, node=0, current_path=[]):
        labels, data = animals_data
        tree = decision_tree.tree_
        left = tree.children_left[node]
        right = tree.children_right[node]
        current_path = current_path + [node]

        if left == right:  # leaf node
            animal_label = decision_tree.classes_[tree.value[node].argmax()]
            yield (current_path, animal_label)
        else:  # decision node
            yield from find_paths(decision_tree, animals_data, node=left, current_path=current_path)
            yield from find_paths(decision_tree, animals_data, node=right, current_path=current_path)

    paths = list(find_paths(decision_tree, animals_data))
    shortest_path = min(paths, key=lambda x: len(x[0]))
    longest_path = max(paths, key=lambda x: len(x[0]))
    return shortest_path, longest_path

# Load animals data as a dictionary
animals_data = load_matrix("awa/predicate-matrix-binary.txt")

# Initialize variables to store the least similar animals and the minimum common features
least_similar_animals = None
min_common_features = float('inf')

# Iterate through all pairs of animals and count common features
for animal1, features1 in animals_data.items():
    for animal2, features2 in animals_data.items():
        if animal1 != animal2:
            common_features = sum(f1 == f2 for f1, f2 in zip(features1, features2))
            if common_features < min_common_features:
                min_common_features = common_features
                least_similar_animals = (animal1, animal2)

# Convert animal labels to their corresponding names
animal1_label, animal2_label = least_similar_animals
animal1_name = labels[int(animal1_label)]
animal2_name = labels[int(animal2_label)]

# Print the names of the least similar animals and the number of common features
print("The two least similar animals are:", (animal1_name, animal2_name))
print("Number of common features:", min_common_features)


def find_paths(decision_tree, animals_data, node=0, current_path=[]):
    labels, data = animals_data
    tree = decision_tree.tree_
    left = tree.children_left[node]
    right = tree.children_right[node]
    current_path = current_path + [node]

    if left == right:  # leaf node
        animal_label = decision_tree.classes_[tree.value[node].argmax()]
        return [(current_path, animal_label)]
    else:  # decision node
        left_paths = find_paths(decision_tree, animals_data, node=left, current_path=current_path)
        right_paths = find_paths(decision_tree, animals_data, node=right, current_path=current_path)
        return left_paths + right_paths

# After fitting the decision tree
paths = find_paths(decision_tree, animals_data)
shortest_path = min(paths, key=lambda x: len(x[0]))
longest_path = max(paths, key=lambda x: len(x[0]))
shortest_questions = len(shortest_path[0]) - 1  # Counting internal nodes only
longest_questions = len(longest_path[0]) - 1  # Counting internal nodes only

# Total number of animals
total_animals = len(labels)

# Average number of questions to guess an animal
average_questions = (sum(len(path[0]) - 1 for path in paths) / total_animals)

print("Shortest path (questions):", shortest_questions)
print("Longest path (questions):", longest_questions)
print("Average questions to guess an animal:", average_questions)


def count(x):
    """return the number of 1's appearing in x"""
    return numpy.sum(x == 1)

# Calculate the number of "yes" answers for each feature
yes_counts = [count(feature_column) for feature_column in data.T]

# Find the index of the feature with the most and fewest "yes" answers
most_yes_index = numpy.argmax(yes_counts)
fewest_yes_index = numpy.argmin(yes_counts)

# Get the names of the features
most_yes_feature = features[most_yes_index]
fewest_yes_feature = features[fewest_yes_index]

# Print the results
print("Feature with the most 'yes' answers:", most_yes_feature)
print("Feature with the fewest 'yes' answers:", fewest_yes_feature)

# Calculate the absolute differences between "yes" and "no" answers for each feature
absolute_diff = [abs(count(data[:, i]) - (N - count(data[:, i]))) for i in range(M)]

# Finding the index of the feature with the smallest absolute difference
balanced_feature_index = min(range(M), key=lambda i: absolute_diff[i])

# Get the name of the balanced feature
balanced_feature = features[balanced_feature_index]

# Print the result
print("Most balanced feature:", balanced_feature)

# Find the index of the feature used in the root node of the decision tree
root_feature_index = numpy.argmin(absolute_diff)

# Get the name of the feature at the root node
root_feature = features[root_feature_index]

# Print the first question at the root node
print("First question at the root node:", root_feature)
