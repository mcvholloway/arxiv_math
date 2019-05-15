def make_stop_words_tags():
  stop_words_tags = ['turn', 'map', 'hold', 'ray', 'infinity',
                'people', 'constructive_way', 'few_example', 'criterion', 'old_solution', '_recent_result',
    'relationship', 'difficulties', 'such_problems', 'basic_understanding', 'same_parameter_values', 'physical_motivations',
              'self', 'attention', 'alternative', 'second-named_author', 'recent_breakthrough_result', 'show', 'previous_result',
                   'our_main_new_result', 'several_related_results', 'same_approach', 'streamlined_manner', 'increases', 'increase', ' certain_algebraic_invariants',
              'implications', 'presence', 'data', 'notice', 'difficulty', 'to_one_correspondence', 'plenty', 'cn', 'constant_c',
              'arguement', 'arguments', 'mild_hypotheses', 'bound', 'bounds', 'assume', 'major_ingredient', 'general_framework', 'time',
              'objective', 'objectives', 'novel', 'relation', 'unified_approach', 'suitable_variant', 'our_results', 'similar_result',
              'rise', 'statement', 'statements', 'correspond', 'corresponds', 'entry', 'entries', 'original_approach', 'current_paper',
              'boundedness', 'function', 'functions', 'symbol', 'symbols', 'who', 'important_ideas', 'main_concern', 'physical_importance',
              'key_concept', 'usual_way', 'newly_developed_logic', 'existing_results', 'exposition', 'minor_changes', 'previous_version',
              'small_corrections', 'research_work', 'novelty', 'several_important_classes', 'special_instances', 'state', 'states',
              'mini-course', 'intuitive_idea', 'old-standing_problem', 'journal', 'open_problem', 'such_formulas', 'special_cases',
              'related_formulas', 'recent_result', 'simple_application', 'enough_and_sufficient_condition', 'several_other_identities',
              'brief_introduction', 'his_formula', 'brussels', 'pqr2003_euroschool', 'employed_mathematical_tools', 
              'additional_condition', 'complete_classification', 'ratio', 'generalises', 'similar_results', 'pqr2003_euroschool',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'short-comings',
              'length', 'positive_answer', 'completely_elementary_way', 'present_writer',
              'generality', 'previous_paper_cite', 'such_sets', 'our_aim', 'unusual_simplification', 'unexpectedly_neat_manner',
              'formulation', 'appropriate_site', 'original_map', 'typical_examples', 'period', 'periods',
              'dimension', 'weight', 'power', 'present_new_geometric_approach', 'proposed scheme', 'intersection', 'interesections',
              'uniqueness', 'inequality', 'metrics','structures', 'brief_note', 'function', 'large_class', 'investigation',
              'proof', 'proofs', 'analogue', 'paper', 'papers', 'result', 'results', 'note', 'notes', '_', 
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
              'article', 'articles','examples', 'simplified_proofs', 'open_question', 'seemingly_new_method',
              'developed_method', 'method', 'methods', 'present_paper', 'solution', 'solutions', 'example', 'examples',
              'part', 'parts', 'case', 'cases', 'our_main_result', 'our_emphasis', 'recent_progress', 'use',
              'exactly_this_question', 'following_interpretation', 'different_kinds', 'various_examples', 
              'failure', 'rich_and_interesting_algebraic_structure', 'first_step', 'search', 'its_action', 'arbitrary_', 
              'remark', 'remarks', 'detail', 'details', 'conjecture', 'conjectures', 'tool', 'tools', 'others',
              'context', 'contexts', 'problem', 'problems', 'above_mentioned_results', 'theorem', 'theorems',
              'our_arguments', 'consequence', 'consequences', 'term', 'terms', 'our_classification', 'fact', 'facts',
              'lecture', 'lectures', 'technique', 'techniques', 'necessary_and_sufficient_conditions', 'good_performance',
              'use', 'uses', 'diagram', 'diagrams', 'application', 'applications', 'past_two_decades', 'special_case', 
              'arguement', 'arguments', 'simple_necessary_and_sufficient_condition', 'condition', 'conditions',
              'analogous_result', 'property', 'properties', 'class', 'classes', 'achieved_result', 'other_components', 'component', 'components',
              'math', 'main_result', 'part', 'parts', 'first_part', 'second_part', 'our_examples', 'form', 'forms', 'construction', 'constructions',
              'first_application', 'consideration', 'considerations', 'explicit_formula', 'explicit_formulas', 'definition', 'definitions',
              'present_authors', 'talk', 'talks', 'theory', 'recent_conjecture', 'our_main_result', 'practical_purpose', 'our_problem',
              'our_theory', 'proposed_algorithm', 'intriguing_novelty', 'point', 'points', 
              '_systematic_and_explicit_way', 'new', 'previous_papers', 'author', 'authors', 'complete_description', 'study', 'studies',
              'general_sufficient_condition', 'main_theorem', 'sufficient_condition', 'necessary_condition',
              'typical_example', 'discussed_problem', 'what', 'novel_approach', 'our_proposed_approach',
              'prior_studies', 'our_simulations', 'advantage', 'advantages', 'last_conjecture', 'nonempty_sum', 'right_value',
              'left_value', 'all_the_elements', 'distinct_elements', 'element', 'elements', 'subset', 'subsets',
              'way', 'ways', 'point', 'points', 'relatively_effective_method', 'their_calculation', 'the_structure',
              'full_set', 'above_result', 'our_result', 'notion', 'notions', 'set', 'sets', 'space', 'spaces', 'respect', 'respects',
              'goal', 'goals', 'value', 'values', 'extended_review', 'particularly_easy_construction', 'background_material', 'recent_work',
              'course', 'affirmative_answer', 'generalization', 'generalizations', 'analogous_results', 'analogous_result', 'our_result',
              'many_important_practical_applications', 'our_best_knowledge', 'last_years', 'a_priori', 'much_attention',
              'body', 'work', 'original_parameter', 'term', 'terms', 'new_procedure', 'unknown', 'unknowns', 'similar_properties',
              'former_paper', 'two_oversights', 'overlooked_results', 'original_paper', 'simple_proof', 'unit', 'units',
              'joint_papers', 'recent_results', 'rigorous_derivation', 'number', 'numbers', 'first_section', 'next_section', 'column', 'columns',
              'paper_studies', 'easy_counter-example', 'open_questions', 'open_question', 'recent_joint-work', 'classical_method',
              'new_technique', 'stronger_ones', 'question', 'questions', 'important_consequences', 'classical_constructions', 
              'quite_recent_subject', 'available_results', 'researcher', 'researchers', 'other_main_ingredient', 'ingredient', 'ingredients',
              'idea', 'ideas', 'rapid_introduction', 'exercise', 'exercises', 'glimps', 'glimpse', 'following_classes', 'following_class',
              '-spaces', 'main_point', 'formula', 'formulas', 'formulae', 'further_questions', 'last_section', 'first_author',
              'previous_work', 'contrast', 'family', 'families', 'best_known_example', 'the_sum', 'contributions', 'contribution',
              'revised_version_mistake', 'other_hand', 'hand', 'affairs', 'lecture_notes', 'current_state',
              'basic_properties', 'setting', 'settings', 'concept', 'concepts', 'existence', 'then_any_model', 'earlier_result_math',
              'upper_and_lower_bounds', 'classification', 'system', 'several_different_classes', 'previous_paper',
              'whole_space', 'improved_versions', 'version', 'versions', 'variable', 'variables', 'two_kinds', 'kind', 'kinds',
              'same_or_longer_length', 'two_distinct_vectors', 'lower_bounds', 'vector', 'vectors', 'excellent_performance', 
              'cite', 'purpose', 'research_project', 'literature', 'framework', 'classification', 'characterization', 'characterizations',
              'concept', 'past_decades', 'recent_reports', 'similar_way', 'same_canonical_role', 'invent', 'blog', 'my_results',
              'converse', 'long-standing_conjecture', 'simpler_proof', 'corollaries', 'long_and_complicated_expressions', 'most_accurate_results',
              'wide_range', 'the_set', 'very_first_results', 'universal', 'processes', 'processe', 'theories', 'model', 'models',
              'well-known_results', 'support', 'second_author', 'last_results', 'great_efforts', 'theoretical_properties', 
              'future_work', 'several_authors', 'one_hand', 'review', 'general_case', 'that', 'parameter', 'parameters', 'approach', 'upper_bound',
              'propose', 'structure', 'main_contribution', 'previous_works', 'announcements', 'more_detailed_look', 'brief_history',
              'newly_apparent_role', 'programs_successes', 'several_sufficient_conditions', 'it', 'second_named_author',
              'similar_features', 'other_contexts', 'emphasis', 'fulfills', 'fulfill', 'reason', 'reasons', 'well_known_fact',
              'equation', 'equations', 'appendix', 'aim', 'size', 'extended_english_abstract',
              'systematic_way', 'general_results', 'approach', 'previous_results', 'his_joint_work', 'his_own_work', 'short_note',
              'first_attempt', 'new_applications', 'more_difficult_problem', 'behavior',
              'alternative_proofs', 'approach', 'merit', 'original_proof', 'several_applications', 'that_equation', 'general_result',
              'law', 'interpretation', 'estimate', 'estimates', 'viewpoint', 'type', 'types', 'short_introduction',
              'novel_estimation_procedure', 'novel_notion', 'new_series', 'new_examples', 'several_properties',
              'very_short_paper', 'same_result', 'arxiv', 'somewhat_stronger_and_more_precise_version', 'first_paper',
              'total_number', 'assumption', 'assumptions', 'important_case', 'main_difficulty', 'settles_question',
              'his_result', 'stronger_conditions', 'partial_positive_answer', 'rather_recent_subject', 'short_proofs', 'new_proof',
              'main_results', 'keywords', '_textit', 'textsc', 'uniqueness_and_existence_results', 'rather_general_setting',
              'right-hand_side', 'above_condition', 'very_explicit_way', 'present_papers', 'second_application',
              'extensive_recent_literature', 'introductory_level_survey', 'previous_preprint', 'insight', 'insights',
              'and_related_questions', 'negative_consistency_results', 'positive_results', 'already_published_results',
              'evaluation', 'depth', 'subclass', 'join', 'multiplicity', 'degree', 'degrees',
             ]
  return stop_words_tags

def make_remove_adjectives():
  remove_adjectives = ['certain', 'new', 'corresponding', 'their', 'our', 'such', 'whose', 'following',
                    'known','different', 'its', 'so-called', 'only', 'namely', 'just', 'same', 'particular',
                    'various', 'interesting', 'given', 'underlying', 'this', 'explicit', 'other', 'celebrated',
                    'respectively', 'associated', 'above', 'many', 'claimed', 'useful', 'that', 'and', 'conjectured',
                    'two', 'efficient', 'his', 'her']
  return remove_adjectives

def make_stop_words():
  stop_words = ['proof', 'first', 'new', 'the', 'a', 'an','certain', 'new', 'corresponding', 'their', 'our', 'such', 'whose', 'following',
                    'known','different', 'its', 'so-called', 'only', 'namely', 'just', 'same', 'particular',
                    'various', 'interesting', 'given', 'underlying', 'this', 'explicit', 'other', 'celebrated',
                    'respectively', 'associated', 'above', 'many', 'claimed', 'useful', 'that', 'conjectured',
                    'two', 'efficient', 'his', 'her', 'several', 'related', 'purely', 'well-known', 'important', 'technical',
             'assumption', 'notion', 'textin', 'doe', 'aforementioned', 'specific', 'nice']
  return stop_words
  return remove_adjectives
