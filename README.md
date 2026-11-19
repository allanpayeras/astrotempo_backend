# Backend of the Astrotempo app

The [Astrotempo](https://app.astrotempo.org/) app is a weather forecast tool focused on astronomical observations.
At the moment, the app is directed to users located in Brazil.
The user will obtain useful meteorological and astronomical information to plan an observation session.

Currently, the tool includes:
- Hourly forecast of cloud cover,
- Near real-time satellite images of cloud presence and their thickness,
- Rise and set times for the Sun and the Moon,
- Moon illumination.

## How it works

Weather forecast data for the Brazilian territory is fetched from [INPE](https://www.gov.br/pt-br/servicos/obter-dados-provenientes-de-modelos-numericos-de-previsao-de-tempo-inpe-pnt).
The data is processed to extract the relevant information for astronomical observations.
The extracted information is stored in a DuckDB database for each point in the grid.

Near real-time meteorological images are obtained from the [GOES-19](https://www.nesdis.noaa.gov/our-satellites/currently-flying/geostationary-satellites) satellite.
The imagery products are processed to create geo tiles that can be served from a static server.

This repository also includes the code of the APIs that serve the processed information to external users and the app [frontend](https://github.com/allanpayeras/astrotempo_flutter_app).
