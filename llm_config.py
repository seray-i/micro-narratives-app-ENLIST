```python
import tomllib

from langchain_core.prompts import PromptTemplate


class LLMConfig:
    """
    Configuration for the micro-narratives app.
    """

    @classmethod
    def from_file(cls, filename):
        with open(filename, "rb") as f:
            config = tomllib.load(f)

        return cls(config)

    def __init__(self, config):
        """
        Initialise configuration from TOML config file.
        """

        # Participant settings
        participant_config = config.get("participant", {})

        self.require_participant_id = participant_config.get(
            "require_participant_id",
            False,
        )

        self.editable_participant_id = participant_config.get(
            "editable_participant_id",
            True,
        )

        self.participant_collection_text = participant_config.get(
            "text",
            "",
        )

        self.require_previous_final_scenario = (
            self.check_if_previous_scenario_required(
                config["collection"]["intro"]
            )
        )

        # General settings
        self.model_name = config.get("model", "gpt-4o")

        # Consent page
        self.intro_and_consent = (
            config["consent"]["intro_and_consent"].strip()
        )

        # Conversation page
        self.questions_intro_prompt_template = (
            PromptTemplate.from_template(
                config["collection"]["intro"].strip()
                + "\n\nLet me know when you're ready!"
            )
        )

        self.questions_prompt_template = (
            self.generate_questions_prompt_template(
                config["collection"]
            )
        )

        self.questions_outro = (
            "Great, I think I got all I need -- but let me double check!"
        )

        # Extraction process
        self.extraction_prompt_template = (
            self.generate_extraction_prompt_template(
                config["summaries"]["questions"]
            )
        )

        self.summary_keys = list(
            config["summaries"]["questions"].keys()
        )

        # Scenario generation
        self.personas = [
            persona.strip()
            for persona in config["summaries"]["personas"].values()
        ]

        self.one_shot = self.generate_one_shot(
            config["example"]
        )

        self.one_shot_conversation = (
            config["example"]["conversation"].strip()
        )

        self.scenario_prompt_template = (
            self.generate_scenario_prompt_template(
                config["summaries"]["questions"]
            )
        )

        self.adaptation_prompt_template = (
            self.generate_adaptation_prompt_template()
        )

    def check_if_previous_scenario_required(self, intro_text):
        """
        Determines whether a previous scenario is required.
        """

        require_previous_scenario = (
            "{previous_scenario}" in intro_text
        )

        if (
            require_previous_scenario
            and not self.require_participant_id
        ):
            raise ValueError(
                "Text of a previous scenario is required, "
                "but participant ID is not tracked. "
                "Set 'require_participant_id' = true in configuration file"
            )

        return require_previous_scenario

    def generate_questions_prompt_template(
        self,
        data_collection,
    ):
        """
        Creates prompt template for collecting answers.
        """

        questions_prompt_text = (
            "{persona}\n\n"
            "Your goal is to gather structured answers to the following questions:\n\n"
            "{questions}\n"
            "Ask each question one at a time.\n"
            "{language_type}\n"
            "Ensure you get at least a basic answer to each question before moving to "
            "the next.\n"
            "Never answer for the human. "
            "If you are unsure what the human meant, ask again. "
            "{topic_restriction}\n"
            "{collection_complete}, stop the conversation and write a single word "
            '"FINISHED".\n\n'
            "Current conversation:\n"
            "{history}\n"
            "Human: {input}\n"
            "AI: "
        )

        return PromptTemplate(
            template=questions_prompt_text,
            input_variables=[
                "history",
                "input",
            ],
            partial_variables={
                "persona": data_collection["persona"],
                "questions": self._generate_question_list(
                    data_collection["questions"]
                ),
                "language_type": data_collection["language_type"],
                "topic_restriction": data_collection["topic_restriction"],
                "collection_complete": (
                    self._generate_collection_complete_text(
                        data_collection["questions"]
                    )
                ),
            },
        )

    def _generate_question_list(self, questions):
        """
        Creates numbered list of questions.
        """

        question_list = ""

        for count, question in enumerate(questions):
            question_list += f"{count + 1}. {question}\n"

        return question_list

    def _generate_collection_complete_text(
        self,
        questions,
    ):
        """
        Creates completion instruction text.
        """

        n_questions = len(questions)

        if n_questions == 1:
            return (
                "Once you have collected an answer to the question"
            )

        return (
            f"Once you have collected answers to all "
            f"{n_questions} questions"
        )

    def generate_extraction_prompt_template(
        self,
        questions,
    ):
        """
        Creates extraction prompt template.
        """

        extraction_prompt_text = (
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the Human answers in the text. "
            "Use only the words and phrases that the text contains. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.\n\n"
            "You will output a JSON with {keys_string} keys.\n\n"
            "{questions}\n"
            "Message to date: {conversation_history}\n\n"
            "Remember, only extract text that is in the messages above "
            "and do not change it."
        )

        return PromptTemplate(
            template=extraction_prompt_text,
            input_variables=[
                "conversation_history",
            ],
            partial_variables={
                "keys_string": self._generate_summary_keys(
                    questions
                ),
                "questions": self._generate_summary_questions(
                    questions
                ),
            },
        )

    def _generate_summary_keys(self, questions):
        """
        Produces comma-separated string of keys.
        """

        keys = list(questions.keys())

        if not keys:
            return ""

        if len(keys) == 1:
            return f"`{keys[0]}`"

        keys_string = ", ".join(
            f"`{key}`"
            for key in keys[:-1]
        )

        keys_string += f", and `{keys[-1]}`"

        return keys_string

    def _generate_summary_questions(self, questions):
        """
        Produces formatted summary questions.
        """

        questions_text = (
            "These correspond to the following question"
            f"{'s' if len(questions) else ''}:\n"
        )

        for count, question in enumerate(
            questions.values()
        ):
            questions_text += (
                f"{count + 1}: {question}\n"
            )

        return questions_text

    def generate_adaptation_prompt_template(self):
        """
        Creates adaptation prompt template.
        """

        return PromptTemplate.from_template(
            "You're a helpful assistant, helping students adapt a scenario "
            "to their liking. The original scenario this student came with:\n\n"
            "Scenario: {scenario}.\n\n"
            "Their current request is {input}.\n\n"
            "Suggest an alternative version of the scenario. "
            "Keep the language and content as similar as possible, "
            "while fulfilling the student's request.\n\n"
            "Return your answer as a JSON file with a single entry "
            "called 'new_scenario'."
        )

    def generate_one_shot(self, example):
        """
        Creates one-shot example.
        """

        return (
            "Example:\n"
            f"{example['conversation'].strip()}\n\n"
            "The scenario based on these responses:\n"
            f"\"{example['scenario'].strip()}\""
        )

    def generate_scenario_prompt_template(
        self,
        questions,
    ):
        """
        Creates scenario generation prompt.
        """

        scenario_prompt_template_text = (
            "{persona}\n\n"
            "{one_shot}\n\n"
            "Your task:\n"
            "Create a scenario based on the following answers:\n\n"
            + self._generate_q_and_a(questions)
            + (
                "\n"
                "Create a scenario based on these responses.\n\n"
                "Your output should be a JSON file with a single entry "
                'called "output_scenario".'
            )
        )

        return PromptTemplate(
            template=scenario_prompt_template_text,
            input_variables=[
                "persona",
                *list(questions.keys()),
            ],
            partial_variables={
                "one_shot": self.one_shot,
            },
        )

    def _generate_q_and_a(self, questions):
        """
        Creates formatted question/answer pairs.
        """

        q_and_a = ""

        for key, question in questions.items():
            q_and_a += (
                f"Question: {question}\n"
            )

            q_and_a += (
                f"Answer: {{{key}}}\n"
            )

        return q_and_a
```
