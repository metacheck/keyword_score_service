import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from keybert import KeyBERT
import aiohttp_cors
from aiohttp import web
from aiohttp.helpers import get_running_loop


routes = web.RouteTableDef()



@routes.post("/test")
async def test(request):
    def task(text):
        res = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words="english")
        res = res+kw_model.extract_keywords(text, keyphrase_ngram_range=(2, 2), stop_words="english")
        res = res+kw_model.extract_keywords(text, keyphrase_ngram_range=(3, 3), stop_words="english")
        return res

    # results: list[
    #     Union[list[tuple[str, float]], list[list[tuple[str, float]]]]] = []
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

    return web.json_response({"status": "OK", "scrape_results": complete_scrape_results})


@routes.get("/test2")
async def test(request):
    return web.json_response({"status": "OK", "closed": True})


async def setup_app(app):
    pass


app = None
thread_pool = None
kw_model = KeyBERT()


def run():
    global app
    global thread_pool
    global kw_model

    thread_pool = ThreadPoolExecutor()
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
    web.run_app(app, port=os.environ.get("PORT",8889))
