import unittest
from langidentification import LangIdentification, LangIdentificationException


class LangIdentificationTestPreprocessText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setting up with original model because it is smaller
        cls.model = LangIdentification(model_type='original')

    def test_01_preprocess_text_str_punct(self):
        """
        Test preprocessing a valid string for whether punctuation is removed
        """
        preprocessed_text = self.model.preprocess_text('This !?,is "some@#|` text.')

        self.assertEqual('This is some text',
                         preprocessed_text)

    def test_02_preprocess_text_str_no_punct(self):
        """
        Test preprocessing a valid string without punctuation but with numbers and spaces to see it is the same
        """
        preprocessed_text = self.model.preprocess_text('This is some text with 7 some 2 numbers ')

        self.assertEqual('This is some text with 7 some 2 numbers ',
                         preprocessed_text)

    def test_03_predict_lang_str_blank(self):
        """
        Test preprocessing a blank string to see if it goes through
        """
        preprocessed_text = self.model.preprocess_text("")

        self.assertEqual("",
                         preprocessed_text)

    def test_04_preprocess_text_list_punct(self):
        """
        Test preprocessing a valid list of strings for whether punctuation is removed
        """
        preprocessed_text = self.model.preprocess_text(['This !?,is "some@#|` text.',
                                                       'There is )*some punctuation('])

        self.assertEqual(['This is some text',
                         'There is some punctuation'],
                         preprocessed_text)

    def test_05_preprocess_text_list_not_all_str(self):
        """
        Test preprocessing a list of objects of which some are not strings to see if error is raised
        """
        with self.assertRaises(LangIdentificationException) as error:
            self.model.preprocess_text(['This !?,is "some@#|` text.',
                                        24])

        self.assertEqual('Not all objects in given input list are strings.',
                         str(error.exception))

    def test_06_preprocess_text_neither_str_not_list(self):
        """
        Test preprocessing an object that is neither a list nor a str to see if error is raised
        """
        with self.assertRaises(LangIdentificationException) as error:
            self.model.preprocess_text({'dict_key': 'dict_value'})

        self.assertEqual('Given text is neither a str nor a list.',
                         str(error.exception))


class LangIdentificationTestPredictLang(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setting up with augmented data model
        cls.model = LangIdentification(model_type='augmented')

    def test_01_test_ta_rom(self):
        """
        Test some romanized formal Tamil
        """
        prediction = self.model.predict_lang('indhiya idhayangalil hockeyai uchathukku kondu sendra singa pengal')

        self.assertEqual('__label__ta-rom',
                         prediction[0][0])

    def test_02_test_ta_rom(self):
        """
        Test some romanized spoken Tamil
        """
        prediction = self.model.predict_lang('naan evlo solliyum kekkaama avan paattukku irukkaan')

        self.assertEqual('__label__ta-rom',
                         prediction[0][0])

    def test_03_test_ml_rom(self):
        """
        Test some romanized Malayalam
        """
        prediction = self.model.predict_lang('athe, athu thanne. athu nallapadi theerkkanam ketto')

        self.assertEqual('__label__ml-rom',
                         prediction[0][0])

    def test_04_test_hi_rom(self):
        """
        Test some romanized Hindi
        """
        prediction = self.model.predict_lang('teesri lehar ke dar ke beech kya yeh school kholne ka sahi waqt hai?')

        self.assertEqual('__label__hi-rom',
                         prediction[0][0])

    def test_04_test_ar_rom(self):
        """
        Test some romanized Arabic
        """
        prediction = self.model.predict_lang('7ala2 bel beit w rayi7 3al she5el w reji3 da7ir ma3 lshabeb w inte?')

        self.assertEqual('__label__ar-rom',
                         prediction[0][0])



