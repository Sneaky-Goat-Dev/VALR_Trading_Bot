import time
import hashlib
import hmac

def sign_request(api_key_secret, timestamp, verb, path, body = ""):
    """Signs the request payload using the api key secret
    api_key_secret - the api key secret
    timestamp - the unix timestamp of this request e.g. int(time.time()*1000)
    verb - Http verb - GET, POST, PUT or DELETE
    path - path excluding host name, e.g. '/v1/withdraw
    body - http request body as a string, optional
    """
    payload = "{}{}{}{}".format(timestamp,verb.upper(),path,body)
    message = bytearray(payload,'utf-8')
    signature = hmac.new( bytearray(api_key_secret,'utf-8'), message, digestmod=hashlib.sha512).hexdigest()
    return signature


