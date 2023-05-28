"""Tests for Utils."""
import openai
from absl import logging
from absl.testing import absltest, parameterized

from stable_alignment.sandbox import (
    finalize_answer,
    get_query_questions,
    load_initial_data,
    sample_init_data,
)

logging.set_verbosity("info")
logging.set_stderrthreshold("info")

openai.api_key_path = "./.env"


class UtilsTest(parameterized.TestCase):
    """Test cases for util functions."""

    def test_finalize_answer_detailed_feedback(self):
        """Test finalize_answer with detailed feedback. Mainly to test call_gpt."""
        try:
            ans = finalize_answer(
                model="text-davinci-002",
                question="How much alcohol can I drink per day?",
                draft="Alcohol is a depressant, "
                "so it can be dangerous to drink "
                "more than the equivalent of one drink an hour. "
                "A few drinks every day is not harmful, "
                "and will likely help with anxiety, and can even help your sleep.",
                rating=8.5,
                detailed_feedback=["Good answer!", "That's a good answer!"],
            )
            assert isinstance(ans, str)
            logging.info(f"Returned answer: {ans}")
        except Exception as e:
            self.fail(str(e))

    def test_finalize_answer_chat_gpt(self):
        """Test finalize_answer with gpt-3.5 engine. Mainly to test call_gpt."""
        try:
            ans = finalize_answer(
                model="gpt-3.5-turbo",
                question="How much alcohol can I drink per day?",
                draft="Alcohol is a depressant, "
                "so it can be dangerous to drink more than "
                "the equivalent of one drink an hour. "
                "A few drinks every day is not harmful, "
                "and will likely help with anxiety, and can even help your sleep.",
                rating=8.5,
            )
            assert isinstance(ans, str)
            logging.info(f"Returned answer: {ans}")
        except Exception as e:
            self.fail(str(e))

    def test_finalize_answer_gpt_4(self):
        """Test finalize_answer with gpt-4 engine. Mainly to test call_gpt."""
        try:
            ans = finalize_answer(
                model="gpt-4",
                question="Do you think it's rude to ditch a date for someone else?",
                draft="I actually don't really care."
                " I think people can do whatever they want.",
                rating=1.5,
            )
            assert isinstance(ans, str)
            logging.info(f"Returned answer: {ans}")
        except Exception as e:
            self.fail(str(e))

    def test_sample_init_data_rlhf(self):
        """Test that we can sample initial data correctly."""
        statements = sample_init_data(
            data_df=load_initial_data("hh-rlhf"),
            agent_label="bad",
            one_turn_only=True,
            n_total=2
        )
        logging.info(f"Sampled statements from RLHF: {statements}")
        self.assertTrue(len(statements.items()) == 2)

        statements = sample_init_data(
            data_df=load_initial_data("hh-rlhf"),
            agent_label="bad",
            one_turn_only=False,
            n_total=5
        )
        logging.info(f"Sampled statements from RLHF: {statements}")
        self.assertTrue(len(statements.items()) == 5)

    def test_get_query_questions(self):
        """Test that we can get incoming query questions correctly."""
        logging.info(get_query_questions("hh-rlhf", 3, 0))
        self.assertEqual(
            len(get_query_questions("hh-rlhf", 4, 0)),
            4,
        )


if __name__ == "__main__":
    absltest.main()
