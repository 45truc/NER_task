from skseq.id_feature import IDFeatures
import string

# ----------
# Replicates the same features as the HMM and adds new handcrafted features
# One for word/tag and tag/tag pair
# ----------

class Extended_Features(IDFeatures):

    def __init__(self, dataset):
        super().__init__(dataset)
        self.stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
            'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 
            'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 
            'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
            'should', 'now'
        }
        self.prepositions = {
            'in', 'on', 'at', 'since', 'for', 'ago', 'before', 'to', 'past', 'until', 'by', 'next', 'within', 'throughout'
        }
        self.punctuation = set(string.punctuation)

        self.prefixes = {
            'geo': ["Mount", "Lake", "River", "San", "New"],
            'gpe': ["Republ", "King", "United", "Sultan", "Comm"],
            'tim': ["Jan", "Feb", "Mar", "Apr", "Mon", "Tues"],
            'org': ["Corp", "Univ", "Assoc", "Org", "Comp"],
            'per': ["Mr", "Ms", "Dr", "Prof", "Sir"],
            'art': ["The", "An", "A", "Book", "Film"],
            'nat': ["Canis", "Felis", "Homo", "Pan", "Rosa"],
            'eve': ["World", "Olymp", "Conf", "Expo", "Summ"]
        }
        
        self.suffixes = {
            'geo': ["land", "ville", "city", "town", "berg"],
            'gpe': ["land", "stan", "ica", "nia", "tan"],
            'tim': ["day", "month", "year", "week", "hour"],
            'org': ["Inc", "Ltd", "Corp", "Group", "Co"],
            'per': ["son", "man", "berg", "stein", "ski"],
            'art': ["Book", "Film", "Story", "Piece", "Work"],
            'nat': ["sapiens", "lupus", "domesticus", "tigris", "rubra"],
            'eve': ["Con", "Meet", "Expo", "Fest", "Summit"]
        }
        
    def add_emission_features(self, sequence, pos, y, features):
        """Add word-tag pair feature."""
        """
        A class that extends the functionality of IDFeatures by adding additional features.

        args:
            sequence (Sequence): The input sequence.
            pos (int): The position in the sequence.
            y (int): The label ID.
            features (list): The list of existing features.

        output:
            list: updated list of features
        """
        x = sequence.x[pos]
        # Get tag name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)

        # Get word name from ID.
        if isinstance(x, str):
            x_name = x
        else:
            x_name = self.dataset.x_dict.get_label_name(x)

        # Generate feature name.
        feat_name = "id:{}::{}".format(x_name, y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)

        ###############################
        ######### OUR FEATURES ########
        ###############################

        # Orthographic features
        if x_name.istitle():
            feat_name = "is_title::{}".format(y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)
                
        if x_name.isupper():
            feat_name = "is_upper::{}".format(y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)
                
        if x_name.islower():
            feat_name = "is_lower::{}".format(y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)
                
        if x_name.isdigit():
            feat_name = "is_digit::{}".format(y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # Check if the word is a stopword
        if x_name.lower() in self.stopwords:
            feat_name = "is_stopword::{}".format(y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # Check if the word is a preposition
        if x_name.lower() in self.prepositions:
            feat_name = "is_preposition::{}".format(y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # Check if the word contains non-ASCII characters
        if any(ord(char) > 127 for char in x_name):
            feat_name = "contains_non_ascii::{}".format(y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # Check if the word contains punctuation
        if any(char in self.punctuation for char in x_name):
            feat_name = "contains_punctuation::{}".format(y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # Check if the word contains hyphens
        if '-' in x_name:
            feat_name = "contains_hyphen::{}".format(y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)
        
        # Word length feature
        feat_name = "word_length:{}::{}".format(len(x_name), y_name)
        feat_id = self.add_feature(feat_name)
        if feat_id != -1:
            features.append(feat_id)

        # Word position in sentence
        if pos == 0:
            feat_name = "position:beginning::{}".format(y_name)
        elif pos == len(sequence.x) - 1:
            feat_name = "position:end::{}".format(y_name)
        else:
            feat_name = "position:middle::{}".format(y_name)
        feat_id = self.add_feature(feat_name)
        if feat_id != -1:
            features.append(feat_id)

        # Common prefixes for entity types
        for prefix in self.prefixes.get(y_name, []):
            if x_name.startswith(prefix):
                feat_name = "prefix_{}::{}".format(prefix, y_name)
                feat_id = self.add_feature(feat_name)
                if feat_id != -1:
                    features.append(feat_id)
                
        # Common suffixes for entity types
        for suffix in self.suffixes.get(y_name, []):
            if x_name.endswith(suffix):
                feat_name = "suffix_{}::{}".format(suffix, y_name)
                feat_id = self.add_feature(feat_name)
                if feat_id != -1:
                    features.append(feat_id)


        return features
