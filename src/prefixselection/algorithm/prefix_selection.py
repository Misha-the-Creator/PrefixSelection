# import json
import os

from .generate import Generation


class PrefixDense:
    def __init__(self,
                 probs_generator: Generation,
                 input_str: str,
                 stop_token: list[str]):
        self.prob_distribution_1 = {}
        self.prob_distribution_2 = {}
        self.prob_distributions = {}
        self.step_matrix = {}
        self.model_to_run = ''
        self.max_steps: int = 10
        self.probs_generator = probs_generator
        self.user_prompt = input_str
        self.generated_text = ""
        self.max_steps = 300
        self.stop_token = (
            [stop_token] if isinstance(stop_token, str) else list(stop_token)
        )
        self.stop_token = [stop_token] if isinstance(stop_token, str) else list(stop_token)
        self.stop_set = set(self.stop_token)

    
    def runpipe(self):
        self.probs_generator.initialize_chat(user_message=self.user_prompt)
        for _ in range(self.max_steps):

            raw = self.probs_generator.generate_pipe(generated_text=self.generated_text)
            clean, stop, _, _ = self._check_eos(raw)
            if stop:
                break

            self.prob_distributions = clean

            selected_text = self.match_chars()
            if not selected_text:
                break

            self.generated_text += selected_text

            cut = self._find_stop(self.generated_text)
            if cut is not None:
                self.generated_text = self.generated_text[:cut]
                break
        return self.generated_text

    
    def count_min_token_len_per_distrib(self, distribs: dict):
        distrib_min_len = {}
        distrib_max_len = {}
        for model_num, distrib in distribs.items():
            distrib_min_len[model_num] = min({len(key) for key in distrib})
            distrib_max_len[model_num] = max({len(key) for key in distrib})
        return distrib_min_len, distrib_max_len


    def find_the_suitest_token(self, 
                               distrib: dict, 
                               char_num: int, 
                               distrib_num: int):
        char_prob = {}
        new_distrib = {}
        for token, probability in distrib.items():
            if char_num >= len(token):
                result = self.probs_generator.generate_pipe(generated_text=self.generated_text + token,
                                                            model_to_run=f"{distrib_num}")
                if isinstance(result, (tuple, list)):
                    result = result[distrib_num]
                for token2, prob2 in result[distrib_num].items():
                    if not token2 or prob2 <= 0 or token2 in self.stop_set:
                        continue
                    ongoing_token = token + token2
                    ongoing_prob = probability*prob2
                    if token not in new_distrib:
                        new_distrib[token] = {token2: ongoing_prob}
                    else:
                        new_distrib[token][token2] = ongoing_prob

                    if char_num < len(ongoing_token):
                        current_char = ongoing_token[char_num]
                        if current_char not in char_prob:
                            char_prob[current_char] = ongoing_prob
                        else:
                            char_prob[current_char] += ongoing_prob
            else:
                current_char = token[char_num]
                char_prob[current_char] = (char_prob.get(current_char, 0.0) + probability)
                new_distrib[token] = (new_distrib.get(token, 0.0) + probability)

        keys_to_drop = [
            key for key, value in new_distrib.items() if isinstance(value, dict)
        ]
        if keys_to_drop:
            future_new_distrib = {}
            for key in keys_to_drop:
                for key2, prob in new_distrib[key].items():
                    joined = key + key2
                    future_new_distrib[joined] = future_new_distrib.get(joined, 0.0) + prob
                del new_distrib[key]
            new_distrib = {**new_distrib, **future_new_distrib}
        return new_distrib, char_prob
    
    def ensemble(self, ensemble_distr: dict):
        token_prob = {}
        for distrib in ensemble_distr.values():
            for token, prob in distrib.items():
                if token not in token_prob:
                    token_prob[token] = prob
                else:
                    token_prob[token] += prob
        
        the_most_popular_token = max(token_prob, key=token_prob.get)
        return the_most_popular_token

    def ensemble_str(self, ensemble_dict: dict, probs_list: list):
        ensembling_probs = {}
        for distr in ensemble_dict.values():
            result = "".join(distr[key] for key in sorted(distr))
            for distrib in probs_list.values():
                for token, prob in distrib.items():
                    if token.startswith(result):
                        if result not in ensembling_probs:
                            ensembling_probs[result] = prob
                        else:
                            ensembling_probs[result] += prob        
        most_likely_token = max(ensembling_probs, key=ensembling_probs.get)
        return most_likely_token

    def _check_eos(self, distributions, votes_ratio=0.5, mass_threshold=0.5):
        n = max(len(distributions), 1)
        votes, eos_mass, clean = 0, 0.0, {}
        for num, distrib in distributions.items():
            if not distrib:
                continue
            if max(distrib, key=distrib.get) in self.stop_set:
                votes += 1
            eos_mass += sum(p for t, p in distrib.items() if t in self.stop_set)
            d = {t: p for t, p in distrib.items() if t not in self.stop_set}
            if d:
                clean[num] = d
        eos_mass /= n
        stop = (votes / n >= votes_ratio) or (eos_mass >= mass_threshold) or (not clean)
        return clean, stop, votes, eos_mass

    def _find_stop(self, text: str):
        pos = [p for p in (text.find(s) for s in self.stop_token) if p != -1]
        return min(pos) if pos else None
   
    def match_chars(self):
        probs_list = self.prob_distributions
        n_models = len(self.probs_generator.llms)
        ensemble_dict = {}
        prefix = ''
        max_char_steps = 128
        char_num = 0
        winning_char = {}
        while char_num < max_char_steps:
            for_suitest = {}
            dict_for_new_distrib = {}
            is_completion = False
            for distrib_num, distrib in probs_list.items():
                new_distrib, char_dict = self.find_the_suitest_token(distrib=distrib,
                                                                     char_num=char_num,
                                                                     distrib_num=distrib_num) 
                for_suitest[distrib_num] = new_distrib
                if distrib != new_distrib:
                    is_completion = True
                    dict_for_new_distrib[distrib_num] = new_distrib
                if char_num not in ensemble_dict:
                    ensemble_dict[char_num] = {distrib_num: char_dict}
                else:
                    ensemble_dict[char_num][distrib_num] = char_dict

            new_prob_list = {} 

            prob_counter = {}
            repeat_counter = {}
            for distr in ensemble_dict.get(char_num, {}).values():
                for token, prob in distr.items():
                    if token not in prob_counter:
                        prob_counter[token] = prob
                        repeat_counter[token] = 1
                    else:
                        prob_counter[token] += prob
                        repeat_counter[token] += 1

            if not prob_counter:
                break


            recalculated_prob = {c: p / n_models for c, p in prob_counter.items()}


            winning_char[char_num] = max(recalculated_prob, key=recalculated_prob.get)
            prefix = "".join(winning_char[key] for key in sorted(winning_char))


            if is_completion:
                for distrib_num, distrib in for_suitest.items():
                    for token, prob in distrib.items():
                        if token.startswith(prefix):
                            if distrib_num not in new_prob_list:
                                new_prob_list[distrib_num] = {token: prob}
                            else:
                                if distrib_num not in new_prob_list:
                                    new_prob_list[distrib_num] = {token: prob}
                                else:
                                    new_prob_list[distrib_num][token] = prob
            else:
                for distrib_num, distrib in probs_list.items():
                    for token, prob in distrib.items():
                        if token.startswith(prefix):
                            if distrib_num not in new_prob_list:
                                new_prob_list[distrib_num] = {token: prob}
                            else:
                                if distrib_num not in new_prob_list:
                                    new_prob_list[distrib_num] = {token: prob}
                                else:
                                    new_prob_list[distrib_num][token] = prob
            ###############################################################
                            
            distrib_schema = []
            for model_num, distrib in new_prob_list.items():
                distrib_schema.append((model_num, distrib))

            ones_counter = 0
            for elem in distrib_schema:
                if len(elem[1]) == 1:
                    ones_counter += 1

            if distrib_schema and ones_counter == len(distrib_schema):
                candidates = [next(iter(d)) for _, d in distrib_schema]
                token = os.path.commonprefix(candidates)
                return token

            char_num += 1
            probs_list = new_prob_list
            if not probs_list:
                break

        return prefix