import json
import os

from generate import Generation


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
        for step in range(self.max_steps):
            print("###############################################################")
            print(f"Шаг: {step}")
            print("Сгенерированное продолжение:")
            print(repr(self.generated_text))

            raw = self.probs_generator.generate_pipe(generated_text=self.generated_text)
            clean, stop, votes, eos_mass = self._check_eos(raw)
            print(f'EOS: голосов {votes}/{len(raw)}, средняя масса {eos_mass:.4f}')
            if stop:
                print('Модели проголосовали за конец генерации — стоп')
                break

            self.prob_distributions = clean          # <-- EOS в матчинг не попадают
            self.pretty_print(self.prob_distributions)

            selected_text = self.match_chars()
            if not selected_text:
                print("match_chars() вернул пустоту — останавливаем генерацию")
                break

            self.generated_text += selected_text
            print(f"Выбранный фрагмент: {selected_text!r}")

            cut = self._find_stop(self.generated_text)   # страховка на случай протечки
            if cut is not None:
                self.generated_text = self.generated_text[:cut]
                print(f"Стоп-строка найдена в тексте, обрезали: {self.generated_text!r}")
                break
        return self.generated_text

    @staticmethod
    def pretty_print(input_dict: dict, string: str = ''):
        print(f'{string}') 
        print(f'{json.dumps(input_dict, indent=2, ensure_ascii=False)}')

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
        self.pretty_print(f'На вход find_the_suitest_token() подано распределение №{distrib_num}')
        self.pretty_print(distrib, 'Само распределение:')
        print(f"Расчет выполняется для индекса {char_num}")
        char_prob = {}
        new_distrib = {}
        for token, probability in distrib.items():
            if char_num >= len(token):
                print(f'Индекс {char_num} выходит за границы токена "{token}" с вероятностью {probability}')
                result = self.probs_generator.generate_pipe(generated_text=self.generated_text + token,
                                                            model_to_run=f"{distrib_num}")
                if isinstance(result, (tuple, list)):
                    result = result[distrib_num]
                self.pretty_print(result, f'Для токена "{token}" с вероятностью {probability} получили такое продолжение:')
                for token2, prob2 in result[distrib_num].items():
                    if not token2 or prob2 <= 0 or token2 in self.stop_set:
                        continue
                    ongoing_token = token + token2
                    ongoing_prob = probability*prob2
                    print(f'Полученный токен "{ongoing_token}", вероятность={ongoing_prob}')
                    if token not in new_distrib:
                        new_distrib[token] = {token2: ongoing_prob}
                    else:
                        new_distrib[token][token2] = ongoing_prob

                    if char_num < len(ongoing_token):
                        current_char = ongoing_token[char_num]
                        self.pretty_print(char_prob, 'На текущий момент')
                        if current_char not in char_prob:
                            char_prob[current_char] = ongoing_prob
                        else:
                            char_prob[current_char] += ongoing_prob
            else:
                current_char = token[char_num]
                char_prob[current_char] = (char_prob.get(current_char, 0.0) + probability)
                new_distrib[token] = (new_distrib.get(token, 0.0) + probability)
            self.pretty_print(char_prob,
                            f'На выходе из цикла подсчета вероятностей по {char_num}-му символу для модели №{distrib_num} получаем следующее:')

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
            self.pretty_print(new_distrib, 'Полученный после комплита словарь, который подет на дальнейшую итерацию: ')
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
            self.pretty_print(distr, 'Распределение в цикле ensemble_str():')
            result = "".join(distr[key] for key in sorted(distr))
            print(f'Результат сложения префиксов: {result}')
            for model_num, distrib in probs_list.items():
                print(f'Для сложения токенов рассматривается распределение модели №{model_num}')
                for token, prob in distrib.items():
                    if token.startswith(result):
                        print(f'Токен {token} начинается с {result}, добавляем его вероятность')
                        if result not in ensembling_probs:
                            ensembling_probs[result] = prob
                        else:
                            ensembling_probs[result] += prob        
        # pprint(f'{ensembling_probs=}')
        most_likely_token = max(ensembling_probs, key=ensembling_probs.get)
        return most_likely_token

    def _check_eos(self, distributions, votes_ratio=0.5, mass_threshold=0.5):
        """Возвращает (распределения без EOS, надо_ли_стопать, голоса, средняя масса EOS)."""
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
        distrib_min_len, distrib_max_len = self.count_min_token_len_per_distrib(distribs=probs_list)
        n_models = len(self.probs_generator.llms)
        ensemble_dict = {}
        self.pretty_print(probs_list,'Распределение до входа в цикл обработки')
        self.pretty_print(distrib_min_len, 'Минимальная длина токена во всех моделях')
        self.pretty_print(distrib_max_len, 'Максимальная длина токена во всех моделях')
        prefix = ''
        max_char_steps = 128
        char_num = 0
        winning_char = {}
        while char_num < max_char_steps:
            print('####################################################################')
            print(f'Номер символа, по которому будет производится расчет: {char_num}')
            self.pretty_print(probs_list, 'Итерация производится по такому распределению:')
            for_suitest = {}
            dict_for_new_distrib = {}
            is_completion = False
            ### Блок определения локальных победителей ###
            for distrib_num, distrib in probs_list.items():
                new_distrib, char_dict = self.find_the_suitest_token(distrib=distrib,
                                                                     char_num=char_num,
                                                                     distrib_num=distrib_num) 
                self.pretty_print(new_distrib, 'После find_the_suitest_token()')
                for_suitest[distrib_num] = new_distrib
                self.pretty_print(new_distrib, 'Новое распределение (возможно, не изменившееся):')
                if distrib != new_distrib:
                    is_completion = True
                    print('Распределения отличаются. Значит, в new_distrib содержатся ключи продолжения токена. Формируем новое распределение для фильтрации')
                    dict_for_new_distrib[distrib_num] = new_distrib
                self.pretty_print(ensemble_dict, f'Словарь распределений символо на {char_num}-ом индексе ДО перезаписи: ')
                if char_num not in ensemble_dict:
                    ensemble_dict[char_num] = {distrib_num: char_dict}
                else:
                    ensemble_dict[char_num][distrib_num] = char_dict
                self.pretty_print(ensemble_dict, f'Словарь распределений символо на {char_num}-ом индексе ПОСЛЕ перезаписи: ')

            new_prob_list = {} 

            self.pretty_print(ensemble_dict, 'Топ символовов по вероятностям у моделей')
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
                print('Ни одна модель не дала символа на этом индексе — выходим')
                break

            self.pretty_print(prob_counter, 'Накопленная вероятность для определения победителя')
            self.pretty_print(repeat_counter, 'Словарь, отражающий число символов в распределениях моделей')

            recalculated_prob = {c: p / n_models for c, p in prob_counter.items()}

            self.pretty_print(recalculated_prob, 'Перерасчет средней вероятности с учетом повторений')

            winning_char[char_num] = max(recalculated_prob, key=recalculated_prob.get)
            print(f'"Победивший" символ — "{max(recalculated_prob, key=recalculated_prob.get)}"')
            self.pretty_print(winning_char, 'Словарь победивших символов')
            prefix = "".join(winning_char[key] for key in sorted(winning_char))
            print(f'Формируемый префикс, которому будем осуществлять фильтрацию токенов: "{prefix}"')


            ### Отбор только тех токенов, которые начинаются с prefix ###
            self.pretty_print(probs_list, 'Реальный слварь, по которому фильтруемся')
            if is_completion:
                self.pretty_print(dict_for_new_distrib, 'Потенциальный словарь для фильтра:')
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
                            

            self.pretty_print(new_prob_list, f'Отфильтрованное распределение с теми токенами, которые начинаются на "{prefix}"')

            distrib_schema = []
            for model_num, distrib in new_prob_list.items():
                distrib_schema.append((model_num, distrib))


            print(f'Сформированная схема (номер_распределения, число_токенов_прошедших_фильтрацию):\n{distrib_schema=}')


            ones_counter = 0
            for elem in distrib_schema:
                if len(elem[1]) == 1:
                    ones_counter += 1

            if distrib_schema and ones_counter == len(distrib_schema):
                candidates = [next(iter(d)) for _, d in distrib_schema]
                token = os.path.commonprefix(candidates)   # e.g. ["Lisa","Lis","Lin"] -> "Li"
                print(f'Токены моделей: {candidates}, общий префикс: {token!r}')
                return token

            char_num += 1
            probs_list = new_prob_list
            if not probs_list:
                print('После фильтрации не осталось токенов — выходим')
                break

        return prefix