from typing import Any
from anthropic.types import MessageParam


import fn

# saves having to rename files, update prompt versions etc
VERSION: int = 8


def generate_dataset() -> list[dict[str, str]]:
    # we need an evaluator
    evaluator = fn.get_evaluator()

    # now a dataset
    dataset = evaluator.generate_dataset(
        # purpose or goal of the prompt
        task_description="Extract topics out of a passage of text from a scholary article into a JSON array of strings",
        # Describe the different inputs that your prompt requires
        prompt_inputs_spec={
            "content": "One paragraph of text from a scholarly journal written in English"
        },
        # file to store dataset
        output_file="b6_dataset.json",
        # finally the number of test cases.num_cases
        # in a prod situation you'd want 50+
        num_cases=3,
    )
    return dataset


def run_evaluator() -> list:
    # we need an evaluator
    evaluator = fn.get_evaluator()

    # now run the results
    results: list = evaluator.run_evaluation(
        run_prompt,
        dataset_file="b6_dataset.json",
        json_output_file=f"b6_{VERSION}_output.json",
        html_output_file=f"b6_{VERSION}_output.html",
        extra_criteria="""
        - contains a JSON array of strings, containing each topic mentioned in the article. 
        - The strings should contain only a topic without any extra commentary.
        - Response should contain the JSON array and nothing else.
        """,
    )
    return results


def versioned_prompt(version: int, prompt_inputs: dict[str, str]) -> str:
    prompts = {
        # 2.0 starter prompt ...
        1: f"""
        What topics are in here?
        
        {prompt_inputs["content"]}
        """,
        # 8.0 now we're adding structured output and a stop sequence on the chat call
        # average 8.0 score but there's a 6.0 in the results :(
        2: f"""
            What topics are in here?
        
        {prompt_inputs["content"]}
        """,
        # 8.3 Lets add in guidelines - and now we're at 8.3
        3: f"""
            What topics are in here?
        
        Guidelines:
        1. Parse the text
        2. Extract major topics, do not include commentary or parenthesis
        3. Insert major topic into your response
        
        {prompt_inputs["content"]}

        """,
        # 8.0 now add in XML - down to 8.0
        4: f"""
            What topics are in here?
        
        Guidelines:
        1. Parse the text
        2. Extract major topics, do not include commentary or parenthesis
        3. Insert major topic into your response
        
        <journal_article>
        {prompt_inputs["content"]}
        </journal_article>

        """,
        # 3.0 - Maybe if I improve my guidelines, I can get it back up.
        # careful what you wish for as it might come true
        # {"major_topics": []} < it did this and was a major validation issue.
        5: f"""
            What topics are in here?
       
        <journal_article>
        {prompt_inputs["content"]}
        </journal_article>
        
        Guidelines:
        1. Parse the text
        2. Extract major topics, do not include commentary or parenthesis
        3. Do not include secondary or minor topics
        4. Insert major topic into your response
        5. Disregard secondary or minor topics



        """,
        # 8.3: refine the guidelines, and do a oneshot ideal and bad example
        6: f"""
            What topics are in here?
       
        <journal_article>
        {prompt_inputs["content"]}
        </journal_article>
        
        Guidelines:
        1. Parse the tex
        2. Extract major topics only do not extract commentary or items in parenthesis
        3. Insert ONLY major topic into your response
        4. You MUST return a list of strings.
        
        <examples>
        
        <ideal_example>
            Here is an ideal example with sample inputs and ideal outputs:
            
            <sample_input>
              This study examines the relationship between remote work adoption and urban housing demand using longitudinal survey data from 2,847 households across metropolitan areas. Our regression analysis reveals a statistically significant negative correlation (r = -0.42, p < 0.001) between full-time remote work eligibility and residential proximity to central business districts, with workers relocating an average of 12.3 miles farther from city centers. These findings have profound implications for urban planning policy, suggesting that zoning regulations and transit infrastructure investments must be reconsidered as traditional commuting patterns dissolve. Additionally, we document secondary effects on commercial real estate valuations and local tax revenues, indicating that policymakers face a complex trade-off between workforce flexibility and municipal fiscal sustainability.
            </sample_input>
            <ideal_output>
                  [
                      "Remote work adoption",
                      "Urban housing demand",
                      "Residential proximity to central business districts",
                      "Worker relocation patterns",
                      "Urban planning policy",
                      "Zoning regulations",
                      "Transit infrastructure",
                      "Commercial real estate valuations",
                      "Local tax revenues",
                      "Municipal fiscal sustainability"
                    ]
            </ideal_output>
            
            It is ideal because all topics are primary, there are no parenthessis and
            the return response is a JSON list of strings ready for ingestion into
            a data pipeline.
            
        </ideal_example>

        
        <bad_example>
            Here is an bad example with sample inputs and bad outputs:
                
            <sample_input>
              This study examines the relationship between remote work adoption and urban housing demand using longitudinal survey data from 2,847 households across metropolitan areas. Our regression analysis reveals a statistically significant negative correlation (r = -0.42, p < 0.001) between full-time remote work eligibility and residential proximity to central business districts, with workers relocating an average of 12.3 miles farther from city centers. These findings have profound implications for urban planning policy, suggesting that zoning regulations and transit infrastructure investments must be reconsidered as traditional commuting patterns dissolve. Additionally, we document secondary effects on commercial real estate valuations and local tax revenues, indicating that policymakers face a complex trade-off between workforce flexibility and municipal fiscal sustainability.
            </sample_input>
            <bad_output>
                  {{
            "topics": [
                      "Remote work adoption",
                      "Urban housing demand",
                      "Residential proximity to central business districts",
                      "Worker relocation patterns",
                      "Urban planning policy",
                      "Zoning regulations",
                      "Transit infrastructure",
                      "Commercial real estate valuations",
                      "Local tax revenues",
                      "Municipal fiscal sustainability"
                    ]
                  }}
            </bad_output>
            
            This output is bad becasue the response is a JSON dictionary and not
            a json list. This would break upstream data pipelines and cause catastrophic
            error.
        </bad_example>  
        
        """,
        # 8.3 add the feedback from report with a 7 score into bad examples
        7: f"""
            What topics are in here?
       
        <journal_article>
        {prompt_inputs["content"]}
        </journal_article>
        
        Guidelines:
        1. Parse the tex
        2. Extract major topics only do not extract commentary or items in parenthesis
        3. Insert ONLY major topic into your response
        4. You MUST return a list of strings.
        
        <examples>
        
        <ideal_example>
            Here is an ideal example with sample inputs and ideal outputs:
            
            <sample_input>
              This study examines the relationship between remote work adoption and urban housing demand using longitudinal survey data from 2,847 households across metropolitan areas. Our regression analysis reveals a statistically significant negative correlation (r = -0.42, p < 0.001) between full-time remote work eligibility and residential proximity to central business districts, with workers relocating an average of 12.3 miles farther from city centers. These findings have profound implications for urban planning policy, suggesting that zoning regulations and transit infrastructure investments must be reconsidered as traditional commuting patterns dissolve. Additionally, we document secondary effects on commercial real estate valuations and local tax revenues, indicating that policymakers face a complex trade-off between workforce flexibility and municipal fiscal sustainability.
            </sample_input>
            <ideal_output>
                  [
                      "Remote work adoption",
                      "Urban housing demand",
                      "Residential proximity to central business districts",
                      "Worker relocation patterns",
                      "Urban planning policy",
                      "Zoning regulations",
                      "Transit infrastructure",
                      "Commercial real estate valuations",
                      "Local tax revenues",
                      "Municipal fiscal sustainability"
                    ]
            </ideal_output>
            
            It is ideal because all topics are primary, there are no parenthessis and
            the return response is a JSON list of strings ready for ingestion into
            a data pipeline.
            
        </ideal_example>

        
        <bad_example>
            Here is an bad example with sample inputs and bad outputs:
                
            <sample_input>
              We synthesized a series of novel copper(II) complexes with N,N'-bis(salicylidene)ethylenediamine ligands through a condensation reaction at 60°C for 4 hours. The resulting compounds were characterized using X-ray crystallography, revealing octahedral coordination geometry around the metal center. UV-vis spectroscopy showed λmax values between 420-450 nm, consistent with d-d transitions. We evaluated the catalytic activity of these complexes in the oxidation of benzyl alcohol to benzaldehyde using H₂O₂ as the oxidant under aerobic conditions. The turnover frequency (TOF) reached 240 h⁻¹ for the most active catalyst, with excellent substrate selectivity and minimal over-oxidation products.
            </sample_input>
            <bad_output>
                  {{
            "topics": [
                      "Remote work adoption",
                      "Urban housing demand",
                      "Residential proximity to central business districts",
                      "Worker relocation patterns",
                      "Urban planning policy",
                      "Zoning regulations",
                      "Transit infrastructure",
                      "Commercial real estate valuations",
                      "Local tax revenues",
                      "Municipal fiscal sustainability"
                    ]
                  }}
            </bad_output>
            
            This output is bad becasue the response is a JSON dictionary and not
            a json list. This would break upstream data pipelines and cause catastrophic
            error.
        </bad_example>  
       
       <bad_example>
            Here is another bad example with sample inputs and bad outputs:
                
            <sample_input>
              We synthesized a series of novel copper(II) complexes with N,N'-bis(salicylidene)ethylenediamine ligands through a condensation reaction at 60°C for 4 hours. The resulting compounds were characterized using X-ray crystallography, revealing octahedral coordination geometry around the metal center. UV-vis spectroscopy showed λmax values between 420-450 nm, consistent with d-d transitions. We evaluated the catalytic activity of these complexes in the oxidation of benzyl alcohol to benzaldehyde using H₂O₂ as the oxidant under aerobic conditions. The turnover frequency (TOF) reached 240 h⁻¹ for the most active catalyst, with excellent substrate selectivity and minimal over-oxidation products.
            </sample_input>
            <bad_output>
                  [
                    "Copper(II) complexes",
                    "N,N'-bis(salicylidene)ethylenediamine ligands",
                    "Condensation reaction",
                    "X-ray crystallography",
                    "Octahedral coordination geometry",
                    "UV-vis spectroscopy",
                    "Catalytic activity",
                    "Oxidation of benzyl alcohol",
                    "Benzaldehyde synthesis",
                    "Hydrogen peroxide oxidant",
                    "Turnover frequency",
                    "Substrate selectivity"
                  ]
                
            </bad_output>
            
            Although the output format is correct, it has a notable deficiency in the secondary criteria by including 'Condensation reaction,' which is explicitly a procedural/methodological element (described as 'condensation reaction at 60°C for 4 hours')
            The solution does well on identifying concrete technical topics and distinguishing chemical compounds, but the inclusion of a procedural step prevents a higher score. The omission of some minor topics like 'd-d transitions' is less critical than the inclusion of procedural content.
        </bad_example>  
       
       

       
        
        </examples>
       
        
        """,
        # 8.3 again - at this point I'm calling it
        8: f"""
            What topics are in here?
       
        <journal_article>
        {prompt_inputs["content"]}
        </journal_article>
        
        Guidelines:
        1. Parse the tex
        2. Extract major topics only do not extract commentary or items in parenthesis
        3. Insert ONLY major topic into your response
        4. You MUST return a list of strings.
        
        <examples>
        
        <ideal_example>
            Here is an ideal example with sample inputs and ideal outputs:
            
            <sample_input>
              This study examines the relationship between remote work adoption and urban housing demand using longitudinal survey data from 2,847 households across metropolitan areas. Our regression analysis reveals a statistically significant negative correlation (r = -0.42, p < 0.001) between full-time remote work eligibility and residential proximity to central business districts, with workers relocating an average of 12.3 miles farther from city centers. These findings have profound implications for urban planning policy, suggesting that zoning regulations and transit infrastructure investments must be reconsidered as traditional commuting patterns dissolve. Additionally, we document secondary effects on commercial real estate valuations and local tax revenues, indicating that policymakers face a complex trade-off between workforce flexibility and municipal fiscal sustainability.
            </sample_input>
            <ideal_output>
                  [
                      "Remote work adoption",
                      "Urban housing demand",
                      "Residential proximity to central business districts",
                      "Worker relocation patterns",
                      "Urban planning policy",
                      "Zoning regulations",
                      "Transit infrastructure",
                      "Commercial real estate valuations",
                      "Local tax revenues",
                      "Municipal fiscal sustainability"
                    ]
            </ideal_output>
            
            It is ideal because all topics are primary, there are no parenthessis and
            the return response is a JSON list of strings ready for ingestion into
            a data pipeline.
            
        </ideal_example>

        
        <bad_example>
            Here is an bad example with sample inputs and bad outputs:
                
            <sample_input>
              We synthesized a series of novel copper(II) complexes with N,N'-bis(salicylidene)ethylenediamine ligands through a condensation reaction at 60°C for 4 hours. The resulting compounds were characterized using X-ray crystallography, revealing octahedral coordination geometry around the metal center. UV-vis spectroscopy showed λmax values between 420-450 nm, consistent with d-d transitions. We evaluated the catalytic activity of these complexes in the oxidation of benzyl alcohol to benzaldehyde using H₂O₂ as the oxidant under aerobic conditions. The turnover frequency (TOF) reached 240 h⁻¹ for the most active catalyst, with excellent substrate selectivity and minimal over-oxidation products.
            </sample_input>
            <bad_output>
                  {{
            "topics": [
                      "Remote work adoption",
                      "Urban housing demand",
                      "Residential proximity to central business districts",
                      "Worker relocation patterns",
                      "Urban planning policy",
                      "Zoning regulations",
                      "Transit infrastructure",
                      "Commercial real estate valuations",
                      "Local tax revenues",
                      "Municipal fiscal sustainability"
                    ]
                  }}
            </bad_output>
            
            This output is bad becasue the response is a JSON dictionary and not
            a json list. This would break upstream data pipelines and cause catastrophic
            error.
        </bad_example>  
       
       <bad_example>
            Here is another bad example with sample inputs and bad outputs:
                
            <sample_input>
              We synthesized a series of novel copper(II) complexes with N,N'-bis(salicylidene)ethylenediamine ligands through a condensation reaction at 60°C for 4 hours. The resulting compounds were characterized using X-ray crystallography, revealing octahedral coordination geometry around the metal center. UV-vis spectroscopy showed λmax values between 420-450 nm, consistent with d-d transitions. We evaluated the catalytic activity of these complexes in the oxidation of benzyl alcohol to benzaldehyde using H₂O₂ as the oxidant under aerobic conditions. The turnover frequency (TOF) reached 240 h⁻¹ for the most active catalyst, with excellent substrate selectivity and minimal over-oxidation products.
            </sample_input>
            <bad_output>
                  [
                    "Copper(II) complexes",
                    "N,N'-bis(salicylidene)ethylenediamine ligands",
                    "Condensation reaction",
                    "X-ray crystallography",
                    "Octahedral coordination geometry",
                    "UV-vis spectroscopy",
                    "Catalytic activity",
                    "Oxidation of benzyl alcohol",
                    "Benzaldehyde synthesis",
                    "Hydrogen peroxide oxidant",
                    "Turnover frequency",
                    "Substrate selectivity"
                  ]
                
            </bad_output>
            
            Although the output format is correct, it has a notable deficiency in the secondary criteria by including 'Condensation reaction,' which is explicitly a procedural/methodological element (described as 'condensation reaction at 60°C for 4 hours')
            The solution does well on identifying concrete technical topics and distinguishing chemical compounds, but the inclusion of a procedural step prevents a higher score. The omission of some minor topics like 'd-d transitions' is less critical than the inclusion of procedural content.
        </bad_example>  
       
       
              <bad_example>
            Here is another bad example with sample inputs and bad outputs:
                
            <sample_input>
      The concept of human agency has long occupied a central position in philosophical discourse, yet its relationship to moral responsibility remains contested. Enlightenment thinkers posited that rational autonomy constituted the foundation of ethical accountability, a view that shaped legal frameworks throughout the nineteenth century. However, historical analysis reveals how social structures and power dynamics constrained individual choice in ways philosophers had not adequately theorized. Contemporary debates between compatibilists and libertarians reflect this tension: while some argue that moral responsibility requires freedom from deterministic causation, others contend that responsibility can coexist with a deterministic universe if agents act according to their own desires and reasoning. This philosophical disagreement carries profound implications for how we understand criminal justice, personal accountability, and the nature of human freedom itself.
            <bad_output>
                  [
 "Human agency",
  "Moral responsibility",
  "Rational autonomy",
  "Ethical accountability",
  "Social structures",
  "Power dynamics",
  "Individual choice",
  "Compatibilism",
  "Libertarianism",
  "Freedom from deterministic causation",
  "Deterministic universe",
  "Criminal justice",
  "Personal accountability",
  "Human freedom"
                  ]
                
            </bad_output>
            
            The solution meets all mandatory requirements: it is a valid JSON array of strings, contains only topics without commentary, and includes nothing else. It successfully identifies core philosophical concepts as specified in the secondary criteria (agency, moral responsibility, autonomy, determinism, compatibilism, libertarianism). The extraction is comprehensive and recognizes overlapping themes across philosophy, ethics, and history. However, there is minor redundancy in how determinism-related concepts are represented, and the inclusion of application domains (criminal justice) alongside theoretical concepts slightly blurs the distinction between main topics and contextual references. These are minor issues that don't violate requirements but represent room for refinement in topic selection precision.
        </bad_example>  
       
        
        </examples>
       
        
        """,
    }

    return prompts[version]


def run_prompt(prompt_inputs: dict[str, Any]) -> str:

    prompt = versioned_prompt(version=VERSION, prompt_inputs=prompt_inputs)

    # create a list of messages
    messages: list[MessageParam] = []

    # add a user message
    fn.add_user_message(messages, prompt)

    fn.add_assistant_message(
        messages,
        "Here are the topics contained in the text as a JSON list```json",
    )

    # and now return the chat
    return fn.chat(messages, stop_sequences=["```"])
