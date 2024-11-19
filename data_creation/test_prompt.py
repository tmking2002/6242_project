import transformers
import torch

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")


prompt = """
    summarise these reddit comments into a coherent summary of their thoughts on donald trump

    Iâ€™m generally conservative but also open minded. I will absolutely not vote for Trump this year though.  Simply on the issue of supporting Ukraine.
    Trump got like 3 million fewer votes so it quite literally wasnâ€™t the will of the people.
    Does MTG speak for you? Does Boebert speak for you? Does DeSantis, or Abbot, or Trump speak for you?
    "**Fat Kim Un:**

    2014: I have Nuke.  
    2016: Hey everyone I am relevant  
    2017: Listen really I have Nuke  
    2018: Yippee I get American Headline SEE!?  
    2019: I have Nuke damn you I shoot  
    2020: See Japan I shoot  
    2021: Nuke Nuke!  
    2022: Me Rikey Trump  
    2023: Need money from China  
    2024: I have Nuke."
    guess we should start killing the millions of americans trump killed with his anti vax approach.  that gift will keep killing for many decades now that he's popularized conspiracy.  magidiots never cared for americans tho
    Thatâ€™s why attempts to legally overthrow the constitution do not rely on the amendment process. For instance, Republicans have been pushing to end birthright citizenship by decree, despite it being enacted by constitutional amendment. If you control the executive part of the government and you donâ€™t care about being perceived as lawful you can do great things. Having learned from his previous errors, a second Trump administration will include people like Vivek Ramawhatever, MTG, Boebert, Gaetz, Nick Fuentes
    "Not surprised based on longstanding accusations of UNRWA schools radicalizing Gazan children with Islamist rhetoric glorifying martyrdom; also the high number of arms including grenades and rocket launchers found stored by Hamas in UNRWA schools shows a blatant disregard for civilian targets (making them a viable military target according to international law.)


    >*Teachers and schools at the UN agency that runs education and social services for Palestinians regularly call to murder Jews, and create teac"
    Yes to every single one of those questions, except the one about the EU, where the guy responsible is Viktor Orban-- who's BFFs with Trump and gets invited to speak at CPAC (the Republicans' main get-together) pretty much every year, so you might as well count him as a Republican, too.
    "> Bibi keeps returning to power. He doesnâ€™t just appear by magic.

    Does Trump speak for you? Does MTG or Boebert? How about Abbott and DeSantis?

    Democracies sometimes have bad actors make their way to leadership positions. That doesn't mean they earned that spot in a way that indicates the support you claim they have.

    > it is filled by actual racists like Ben-Gvir.

    As I stated: nearly 80% of Israelis want Netanyahu and his cronies (Ben-Gvir and Smotrich included) gone and some even want them impr"
    "It's more of an economic cold war rather than a hot war. This has been going on for some time.


    The U.S is already doing everything it can to limit China's growth without calling it sanctions. New restrictionsâ€”including on exports, imports, direct investment, and financial securities.


    > Like [shutting down Huawei and their 5G network](http://www.wikipedia.org/wiki/Criticism_of_Huawei), removing Huawei's position as the second-largest smartphone maker in the world.


    > An [extension of Trump-era "

"""

prompt = """
    Hello!
"""

result = pipeline(prompt, max_length=1000, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, num_return_sequences=1)

print(result[0]['generated_text'])