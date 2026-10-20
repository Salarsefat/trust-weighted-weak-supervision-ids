##! twws_flows.zeek -- per-direction flow metadata for the TWWS-IDS pipeline.
##!
##! Run as:   zeek -C -r <capture> <path>/twws_flows.zeek
##!
##! Writes ``twws_flows.log`` with one line per connection.  Besides the connection
##! summary of conn.log (state, history, per-direction packet and IP-byte counters)
##! it records, separately for the originator ("o_") and responder ("r_") direction:
##!
##!   * first and last packet time                       (o_first, o_last, ...)
##!   * packets seen by the per-packet handlers          (o_seen)
##!   * TCP SYN/FIN/RST/PSH/ACK/URG packet counts        (o_syn, ...)
##!   * payload bytes announced by the headers           (o_pay)
##!   * payload bytes actually present in the capture    (o_cap)
##!   * Shannon entropy (bits/byte) of the first
##!     ``TWWS::entropy_max_bytes`` captured payload bytes (o_ent) and the number of
##!     bytes it was computed over                       (o_ent_n)
##!
##! and two TLS markers taken from handshake metadata only:
##!
##!   * tls              a ServerHello was observed on the connection
##!   * tls_self_signed  the leaf certificate has identical subject and issuer
##!                      (unset when no certificate is visible, e.g. TLS 1.3)
##!
##! The script loads ``policy/misc/loaded-scripts`` so that every run also writes
##! ``loaded_scripts.log`` (the exact set of Zeek scripts in effect).
##!
##! Compatible with Zeek 5.x - 8.x (uses only long-standing events and BiFs).

@load base/protocols/conn
@load base/protocols/ssl
@load base/files/x509
@load policy/misc/loaded-scripts

module TWWS;

export {
	redef enum Log::ID += { LOG };

	## Number of payload bytes per direction over which entropy is computed.
	option entropy_max_bytes: count = 4096;

	type Info: record {
		ts:             time            &log;
		uid:            string          &log;
		orig_h:         addr            &log;
		orig_p:         port            &log;
		resp_h:         addr            &log;
		resp_p:         port            &log;
		proto:          transport_proto &log;
		duration:       interval        &log &optional;
		conn_state:     string          &log &optional;
		history:        string          &log &optional;
		orig_pkts:      count           &log &optional;
		orig_ip_bytes:  count           &log &optional;
		resp_pkts:      count           &log &optional;
		resp_ip_bytes:  count           &log &optional;

		o_first:        time            &log &optional;
		o_last:         time            &log &optional;
		r_first:        time            &log &optional;
		r_last:         time            &log &optional;
		o_seen:         count           &log &default=0;
		r_seen:         count           &log &default=0;

		o_syn:          count           &log &default=0;
		o_fin:          count           &log &default=0;
		o_rst:          count           &log &default=0;
		o_psh:          count           &log &default=0;
		o_ack:          count           &log &default=0;
		o_urg:          count           &log &default=0;
		r_syn:          count           &log &default=0;
		r_fin:          count           &log &default=0;
		r_rst:          count           &log &default=0;
		r_psh:          count           &log &default=0;
		r_ack:          count           &log &default=0;
		r_urg:          count           &log &default=0;

		o_pay:          count           &log &default=0;
		o_cap:          count           &log &default=0;
		r_pay:          count           &log &default=0;
		r_cap:          count           &log &default=0;

		o_ent:          double          &log &optional;
		o_ent_n:        count           &log &default=0;
		r_ent:          double          &log &optional;
		r_ent_n:        count           &log &default=0;

		tls:             bool           &log &default=F;
		tls_self_signed: bool           &log &optional;
	};
}

redef record connection += {
	twws:    Info               &optional;
	twws_oe: opaque of entropy  &optional;
	twws_re: opaque of entropy  &optional;
};

# Deliver the payload of every UDP datagram to udp_contents.
redef udp_content_deliver_all_orig = T;
redef udp_content_deliver_all_resp = T;

event zeek_init() &priority=5
	{
	Log::create_stream(TWWS::LOG, [$columns=Info, $path="twws_flows"]);
	}

function state(c: connection): Info
	{
	if ( ! c?$twws )
		c$twws = Info($ts=c$start_time, $uid=c$uid,
		              $orig_h=c$id$orig_h, $orig_p=c$id$orig_p,
		              $resp_h=c$id$resp_h, $resp_p=c$id$resp_p,
		              $proto=get_port_transport_proto(c$id$resp_p));
	return c$twws;
	}

function account(c: connection, is_orig: bool, len: count, payload: string)
	{
	local s = state(c);
	local now = network_time();
	local n = |payload|;

	if ( is_orig )
		{
		if ( ! s?$o_first )
			s$o_first = now;
		s$o_last = now;
		++s$o_seen;
		s$o_pay += len;
		s$o_cap += n;
		if ( n > 0 && s$o_ent_n < entropy_max_bytes )
			{
			local room_o = entropy_max_bytes - s$o_ent_n;
			if ( ! c?$twws_oe )
				c$twws_oe = entropy_test_init();
			if ( n <= room_o )
				{
				entropy_test_add(c$twws_oe, payload);
				s$o_ent_n += n;
				}
			else
				{
				entropy_test_add(c$twws_oe, payload[0:room_o]);
				s$o_ent_n += room_o;
				}
			}
		}
	else
		{
		if ( ! s?$r_first )
			s$r_first = now;
		s$r_last = now;
		++s$r_seen;
		s$r_pay += len;
		s$r_cap += n;
		if ( n > 0 && s$r_ent_n < entropy_max_bytes )
			{
			local room_r = entropy_max_bytes - s$r_ent_n;
			if ( ! c?$twws_re )
				c$twws_re = entropy_test_init();
			if ( n <= room_r )
				{
				entropy_test_add(c$twws_re, payload);
				s$r_ent_n += n;
				}
			else
				{
				entropy_test_add(c$twws_re, payload[0:room_r]);
				s$r_ent_n += room_r;
				}
			}
		}
	}

event tcp_packet(c: connection, is_orig: bool, flags: string, seq: count, ack: count, len: count, payload: string)
	{
	account(c, is_orig, len, payload);
	local s = c$twws;
	if ( is_orig )
		{
		if ( "S" in flags ) ++s$o_syn;
		if ( "F" in flags ) ++s$o_fin;
		if ( "R" in flags ) ++s$o_rst;
		if ( "P" in flags ) ++s$o_psh;
		if ( "A" in flags ) ++s$o_ack;
		if ( "U" in flags ) ++s$o_urg;
		}
	else
		{
		if ( "S" in flags ) ++s$r_syn;
		if ( "F" in flags ) ++s$r_fin;
		if ( "R" in flags ) ++s$r_rst;
		if ( "P" in flags ) ++s$r_psh;
		if ( "A" in flags ) ++s$r_ack;
		if ( "U" in flags ) ++s$r_urg;
		}
	}

event udp_contents(u: connection, is_orig: bool, contents: string)
	{
	account(u, is_orig, |contents|, contents);
	}

event ssl_server_hello(c: connection, version: count, record_version: count, possible_ts: time, server_random: string, session_id: string, cipher: count, comp_method: count)
	{
	state(c)$tls = T;
	}

# Runs after Conn has filled c$conn (priority 5) and before the SSL removal hook (-3).
event connection_state_remove(c: connection)
	{
	local s = state(c);

	if ( c?$conn )
		{
		if ( c$conn?$duration )      s$duration = c$conn$duration;
		if ( c$conn?$conn_state )    s$conn_state = c$conn$conn_state;
		if ( c$conn?$history )       s$history = c$conn$history;
		if ( c$conn?$orig_pkts )     s$orig_pkts = c$conn$orig_pkts;
		if ( c$conn?$orig_ip_bytes ) s$orig_ip_bytes = c$conn$orig_ip_bytes;
		if ( c$conn?$resp_pkts )     s$resp_pkts = c$conn$resp_pkts;
		if ( c$conn?$resp_ip_bytes ) s$resp_ip_bytes = c$conn$resp_ip_bytes;
		s$proto = c$conn$proto;
		}

	if ( c?$twws_oe && s$o_ent_n > 0 )
		s$o_ent = entropy_test_finish(c$twws_oe)$entropy;
	if ( c?$twws_re && s$r_ent_n > 0 )
		s$r_ent = entropy_test_finish(c$twws_re)$entropy;

	if ( c?$ssl && c$ssl?$cert_chain && |c$ssl$cert_chain| > 0 && c$ssl$cert_chain[0]?$x509 )
		{
		local cert = c$ssl$cert_chain[0]$x509$certificate;
		s$tls_self_signed = ( cert$subject == cert$issuer );
		}

	Log::write(TWWS::LOG, s);
	}
