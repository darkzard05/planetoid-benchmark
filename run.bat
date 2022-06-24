@echo start

python planetoid.py --dataset=Cora --split=public --model=APPNP
python planetoid.py --dataset=Cora --split=full --model=APPNP
python planetoid.py --dataset=Cora --split=random --model=APPNP

python planetoid.py --dataset=Cora --split=public --model=APPNP --optimizer=NAdam
python planetoid.py --dataset=Cora --split=full --model=APPNP --optimizer=NAdam
python planetoid.py --dataset=Cora --split=random --model=APPNP --optimizer=NAdam

python planetoid.py --dataset=Cora --split=public --model=SGConv
python planetoid.py --dataset=Cora --split=full --model=SGConv
python planetoid.py --dataset=Cora --split=random --model=SGConv

python planetoid.py --dataset=Cora --split=public --model=SGConv --optimizer=NAdam
python planetoid.py --dataset=Cora --split=full --model=SGConv --optimizer=NAdam
python planetoid.py --dataset=Cora --split=random --model=SGConv --optimizer=NAdam

python planetoid.py --dataset=Cora --split=public --model=SplineConv
python planetoid.py --dataset=Cora --split=full --model=SplineConv
python planetoid.py --dataset=Cora --split=random --model=SplineConv

python planetoid.py --dataset=Cora --split=public --model=SplineConv --optimizer=NAdam
python planetoid.py --dataset=Cora --split=full --model=SplineConv --optimizer=NAdam
python planetoid.py --dataset=Cora --split=random --model=SplineConv --optimizer=NAdam

python planetoid.py --dataset=PubMed --split=public --model=APPNP
python planetoid.py --dataset=PubMed --split=full --model=APPNP
python planetoid.py --dataset=PubMed --split=random --model=APPNP

python planetoid.py --dataset=PubMed --split=public --model=APPNP --optimizer=NAdam
python planetoid.py --dataset=PubMed --split=full --model=APPNP --optimizer=NAdam
python planetoid.py --dataset=PubMed --split=random --model=APPNP --optimizer=NAdam

python planetoid.py --dataset=PubMed --split=public --model=SGConv
python planetoid.py --dataset=PubMed --split=full --model=SGConv
python planetoid.py --dataset=PubMed --split=random --model=SGConv

python planetoid.py --dataset=PubMed --split=public --model=SGConv --optimizer=NAdam
python planetoid.py --dataset=PubMed --split=full --model=SGConv --optimizer=NAdam
python planetoid.py --dataset=PubMed --split=random --model=SGConv --optimizer=NAdam

python planetoid.py --dataset=PubMed --split=public --model=SplineConv
python planetoid.py --dataset=PubMed --split=full --model=SplineConv
python planetoid.py --dataset=PubMed --split=random --model=SplineConv

python planetoid.py --dataset=PubMed --split=public --model=SplineConv --optimizer=NAdam
python planetoid.py --dataset=PubMed --split=full --model=SplineConv --optimizer=NAdam
python planetoid.py --dataset=PubMed --split=random --model=SplineConv --optimizer=NAdam

python planetoid.py --dataset=CiteSeer --split=public --model=APPNP
python planetoid.py --dataset=CiteSeer --split=full --model=APPNP
python planetoid.py --dataset=CiteSeer --split=random --model=APPNP

python planetoid.py --dataset=CiteSeer --split=public --model=APPNP --optimizer=NAdam
python planetoid.py --dataset=CiteSeer --split=full --model=APPNP --optimizer=NAdam
python planetoid.py --dataset=CiteSeer --split=random --model=APPNP --optimizer=NAdam

python planetoid.py --dataset=CiteSeer --split=public --model=SGConv
python planetoid.py --dataset=CiteSeer --split=full --model=SGConv
python planetoid.py --dataset=CiteSeer --split=random --model=SGConv

python planetoid.py --dataset=CiteSeer --split=public --model=SGConv --optimizer=NAdam
python planetoid.py --dataset=CiteSeer --split=full --model=SGConv --optimizer=NAdam
python planetoid.py --dataset=CiteSeer --split=random --model=SGConv --optimizer=NAdam

python planetoid.py --dataset=CiteSeer --split=public --model=SplineConv
python planetoid.py --dataset=CiteSeer --split=full --model=SplineConv
python planetoid.py --dataset=CiteSeer --split=random --model=SplineConv

python planetoid.py --dataset=CiteSeer --split=public --model=SplineConv --optimizer=NAdam
python planetoid.py --dataset=CiteSeer --split=full --model=SplineConv --optimizer=NAdam
python planetoid.py --dataset=CiteSeer --split=random --model=SplineConv --optimizer=NAdam

@echo done