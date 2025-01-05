profile_mapping = {
    "nan": "nan",
    "0.0": "nan",
    "1.0": "flat",
    "2.0": "hilly",
    "3.0": "mountainous",
    "4.0": "high mountains",
    "5.0": "uphill finish",
}

season_mapping = {0: "Fall", 1: "Spring", 2: "Summer", 3: "Winter"}

continent_mapping = {
    0: "Africa",
    1: "Asia",
    2: "Europe",
    3: "North America",
    4: "Oceania",
    5: "South America",
}

def rescale_rule(
    rules,
    means,
    stds,
    scaled_cols=["points", "BMI", "cyclist_age", "length", "startlist_quality"],
):
    for rule in rules["premise"]:
        if rule["att"] in scaled_cols:
            idx = scaled_cols.index(rule["att"])
            rule["thr"] = rule["thr"] * stds[idx] + means[idx]

    return rules


def translate_rule(rules):
    cat_cols = ["season", "continent", "profile"]
    dict_cols = {
        "season": False,
        "continent": False,
        "profile": False,
    }
    dict_counters = {
        "season": 0,
        "continent": 0,
        "profile": 0,
    }
    dict_mapping = {
        "season": season_mapping,
        "continent": continent_mapping,
        "profile": profile_mapping,
    }
    for rule in rules["premise"]:
        for cat in cat_cols:
            if cat in rule["att"]:
                # get respective mapping
                mapping = dict_mapping[cat]

                # rename feature to have clearer explanation
                cat_num = rule["att"].split("_")[1]
                if not cat == "profile":

                    rule["att"] = f"is_{cat}_{mapping[int(cat_num)]}"
                else:
                    rule["att"] = f"is_{cat}_{mapping[cat_num]}"

                if "<" in rule["op"]:
                    dict_counters[cat] += 1
                    rule["thr"] = 0
                else:
                    dict_cols[cat] = True
                    rule["thr"] = 1
                rule["op"] = "=="

    for cat in cat_cols:
        if dict_cols[cat]:
            # remove prome premise all rules containing cat that have thr = 0
            rules["premise"] = [
                rule
                for rule in rules["premise"]
                if not (cat in rule["att"] and rule["thr"] == 0)
            ]

        elif dict_counters[cat] == len(dict_mapping[cat]) - 1:
            # check what rules are present for the category, remove them, and put the remaining rule with thr = 1
            cat_rules = [rule for rule in rules["premise"] if cat in rule["att"]]
            cats_found = [rule["att"].split("_")[-1] for rule in cat_rules]
            missing_cat = list(set(dict_mapping[cat]) - set(cats_found))[0]

            rules["premise"] = [
                rule for rule in rules["premise"] if not (cat in rule["att"])
            ]
            rules["premise"].append(
                {"att": f"is_{cat}_{missing_cat}", "op": "==", "thr": 1}
            )

    return rules


def rescale_rule_cf(
    rules_cf,
    means,
    stds,
    scaled_cols=["points", "BMI", "cyclist_age", "length", "startlist_quality"],
):
    for rules in rules_cf:
        for rule in rules["premise"]:
            if rule["att"] in scaled_cols:
                idx = scaled_cols.index(rule["att"])
                rule["thr"] = rule["thr"] * stds[idx] + means[idx]

    return rules_cf


def translate_rule_cf(rules_cf):
    for rules in rules_cf:
        cat_cols = ["season", "continent", "profile"]
        dict_cols = {
            "season": False,
            "continent": False,
            "profile": False,
        }
        dict_counters = {
            "season": 0,
            "continent": 0,
            "profile": 0,
        }
        dict_mapping = {
            "season": season_mapping,
            "continent": continent_mapping,
            "profile": profile_mapping,
        }
        for rule in rules["premise"]:
            for cat in cat_cols:
                if cat in rule["att"]:
                    # get respective mapping
                    mapping = dict_mapping[cat]

                    # rename feature to have clearer explanation
                    cat_num = rule["att"].split("_")[1]
                    if not cat == "profile":

                        rule["att"] = f"is_{cat}_{mapping[int(cat_num)]}"
                    else:
                        rule["att"] = f"is_{cat}_{mapping[cat_num]}"

                    if "<" in rule["op"]:
                        dict_counters[cat] += 1
                        rule["thr"] = 0
                    else:
                        dict_cols[cat] = True
                        rule["thr"] = 1
                    rule["op"] = "=="

        for cat in cat_cols:
            if dict_cols[cat]:
                # remove prome premise all rules containing cat that have thr = 0
                rules["premise"] = [
                    rule
                    for rule in rules["premise"]
                    if not (cat in rule["att"] and rule["thr"] == 0)
                ]

            elif dict_counters[cat] == len(dict_mapping[cat]) - 1:
                # check what rules are present for the category, remove them, and put the remaining rule with thr = 1
                cat_rules = [rule for rule in rules["premise"] if cat in rule["att"]]
                cats_found = [rule["att"].split("_")[-1] for rule in cat_rules]
                missing_cat = list(set(dict_mapping[cat]) - set(cats_found))[0]

                rules["premise"] = [
                    rule for rule in rules["premise"] if not (cat in rule["att"])
                ]
                rules["premise"].append(
                    {"att": f"is_{cat}_{missing_cat}", "op": "==", "thr": 1}
                )

    return rules
