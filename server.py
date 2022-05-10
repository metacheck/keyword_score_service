import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Tuple

import aiohttp_cors
from aiohttp import web
from aiohttp.helpers import get_running_loop
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from yake import yake

routes = web.RouteTableDef()
counter = 0


@routes.post("/test")
async def test(request):
    def task(text):

        # kws: List[Tuple[str, float]] = kw_extractor.extract_keywords(text)
        # seed = list(map(lambda x: str(x[0]), kws))

        res = kw_model.extract_keywords(text, keyphrase_ngram_range=(2, 3), stop_words="english", top_n=7,)
        # res = res + kw_model.extract_keywords(text, keyphrase_ngram_range=(3, 3), stop_words="english")
        return res

    # results: list[
    #     Union[list[tuple[str, float]], list[list[tuple[str, float]]]]] = []

    start = time.time()

    scrape_results = (await request.json())
    scrape_results = scrape_results["scrape_results"]

    tasks = []
    for sres in scrape_results:
        tasks.append(get_running_loop().run_in_executor(thread_pool, lambda: task(text=sres["text"])))
    results = await asyncio.gather(*tasks)
    keyword_result = list(map(lambda x: dict(x), results))

    complete_scrape_results = []
    index = 0
    for scrap_res in scrape_results:
        dic = dict(scrap_res)
        dic.update({"keyword_score": keyword_result[index]})
        complete_scrape_results.append(dic)
        index += 1

    end = time.time()
    global counter
    counter += 1
    print("current ", counter)
    return web.json_response({"status": "OK", "execution_time": end - start, "scrape_results": complete_scrape_results})


@routes.get("/test2")
async def test(request):
    return web.json_response({"status": "OK", "closed": True})


async def setup_app(app):
    pass


app = None
thread_pool = None
kw_model = None
kw_extractor = None


async def run():
    global app
    global kw_extractor
    global thread_pool
    global kw_model
    sentence_model = SentenceTransformer(os.path.abspath("") + "/storage/all-MiniLM-L6-v2/")
    kw_model = KeyBERT(model=sentence_model)
    kw_extractor = yake.KeywordExtractor(dedupLim=0.9,
                                         dedupFunc='seqm', top=10)

    thread_pool = ThreadPoolExecutor(max_workers=50)
    app = web.Application()
    app.on_startup.append(setup_app)
    cors = aiohttp_cors.setup(
        app,
        defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*",
            )
        },
    )
    app.add_routes(routes)
    for route in list(app.router.routes()):
        cors.add(route)
    return app


async def serve():
    return run()


if __name__ == "__main__":
    app = run()
    web.run_app(app, port=os.environ.get("PORT", 9002))
